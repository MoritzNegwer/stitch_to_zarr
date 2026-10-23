#!/usr/bin/env python3
import numpy as np
import argparse
import json
import os
import re
import shutil
import tempfile
import time
import math
import zarr
import imagej
import scyjava
from pathlib import Path
from itertools import product
from skimage import measure
from dask import delayed, compute 


def log(message):
    print('[%s] %s' % (time.strftime('%H:%M:%S'), message), flush=True)


def resolve_path(value, base):
    path = Path(value).expanduser()
    return (base / path).resolve() if not path.is_absolute() else path.resolve()


def load_config(filename, stage_channel=None):
    filename = Path(filename).resolve()
    with filename.open(encoding='utf-8') as stream:
        config = json.load(stream)
    base = filename.parent
    reference_name = config.get('threshold_reference_channel')
    if reference_name:
        references = [c for c in config['channels'] if c['name'] == reference_name]
        if len(references) != 1:
            raise ValueError('threshold_reference_channel must name exactly one configured channel')
        reference = references[0]
    config['fiji_dir'] = str(resolve_path(config['fiji_dir'], base))
    if not Path(config['fiji_dir']).is_dir():
        raise ValueError('fiji_dir must point to an existing Fiji installation')
    channels = config['channels'] if stage_channel is None else [config['channels'][stage_channel]]
    if not channels:
        raise ValueError('Configure at least one channel')
    used_names = set()
    outputs = []
    for channel in channels:
        name = channel['name']
        if not re.fullmatch(r'[A-Za-z0-9_-]+', name) or name in used_names:
            raise ValueError('Channel names must be unique and contain letters, digits, _ or -')
        used_names.add(name)
        tile_config = Path(config['tile_config'])
        if not tile_config.is_file() or not re.search(r'Z\d{4}(?!\d)', tile_config.read_text()):
            raise ValueError('Tile configuration missing or has no Z#### references: ' + str(tile_config))
        for key in ('zarr_output',):
            output = Path(channel[key])
            if output.exists():
                raise ValueError('Choose a new output path; already exists: ' + str(output))
            outputs.append(output)
    for index, output in enumerate(outputs):
        for other in outputs[index + 1:]:
            if output == other or output in other.parents or other in output.parents:
                raise ValueError('Output paths must be distinct and must not contain one another')
    for key, default in [('voxel_size', [6.0, 1.62, 1.62]), ('chunk_shape', [16, 64, 64])]:
        value = config.setdefault(key, default)
        if len(value) != 3 or any(v <= 0 for v in value):
            raise ValueError(key + ' needs three positive values in z, y, x order')
    if any(type(v) is not int for v in config['chunk_shape']):
        raise ValueError('chunk_shape values must be integers')
    if type(config.setdefault('compute_overlap', False)) is not bool:
        raise ValueError('compute_overlap must be true or false')
    return config


def plane_pixels(image):
    """convert Fiji ImagePlus into np.array with y,x order and the correct numpy dtype"""
    if image.getStackSize() != 1 or image.getBitDepth() not in (8, 16, 32):
        raise ValueError('Expected one grayscale stitched plane (8, 16 or 32 bit)')
    bits = image.getBitDepth()
    dtype = {8: np.uint8, 16: np.uint16, 32: np.float32}[bits]
    pixels = np.asarray(image.getProcessor().getPixels(), dtype=dtype)

    return pixels.reshape(image.getHeight(), image.getWidth()).copy()


def level_metadata(config, channel, levels):
    """produce zarr metadata"""
    return {'version': '0.5', 'multiscales': [{
        'name': channel['name'],
        'axes': [{'name': axis, 'type': 'space', 'unit': 'micrometer'} for axis in ('z', 'y', 'x')],
        'datasets': [{'path': str(level), 'coordinateTransformations': [
            {'type': 'scale', 'scale': [float(v) * (1 if axis == 0 else 2 ** level)
                                      for axis, v in enumerate(config['voxel_size'])]},
            {'type': 'translation', 'translation': [float(v) / 2 for v in config['voxel_size']]}
        ]} for level in levels],
        'metadata': {'description': 'Maximum of 1x2x2 blocks at each downsample level', 'method': 'np.max'}
    }]}


def create_groups (config, channel, pixels=None):
    """creates zarr groups with arrays stored on disk, one per channel (+ mask)"""
    
    #create zarr directories
    Path(channel['zarr_output']).parent.mkdir(parents=True, exist_ok=True)
    log('Creating OME Zarr for %s: %s' % (channel['name'], channel['zarr_output']))

    chunks = config["chunk_shape"]   
    
    #first, use a single z-plane as chunk. Rechunk to smaller 3D blocks later 
    zplane_chunks = (1,
        chunks[1] * math.ceil(pixels.shape[0] / chunks[1]),
        chunks[2] * math.ceil(pixels.shape[1] / chunks[2]))

    #padding logic for the z-axis: Z-thickness must be a multiple of chunks[0]
    padded_z_extent = chunks[0] * math.ceil((config['stop_plane'] - config['start_plane']) / chunks[0])
    
    #determine shape
    shape = (padded_z_extent, zplane_chunks[1], zplane_chunks[2])

    #create storage group for the current channel
    group = zarr.open_group(channel['zarr_output'], mode='w-', zarr_format=3)
    group.create_array(
        name='0',
        shape=shape,
        chunks=zplane_chunks,
        shards=None,
        dtype=pixels.dtype,
        dimension_names=('z', 'y', 'x'),
        compressors=[zarr.codecs.BloscCodec(cname='lz4', clevel=5)]
    )
    group.attrs['ome'] = level_metadata(config, channel, [0])

    log('Level 0 ready: shape=%s, dtype=%s, chunks=%s' % (group['0'].shape, group['0'].dtype, group['0'].chunks))

    #extra: If this is for the reference channel, also create a separate mask group 
    if channel['name'] == config.get('threshold_reference_channel'):

        mask_path = str(channel['zarr_output']) + '.threshold-mask.zarr'

        mask = zarr.open_group(mask_path, mode='w-', zarr_format=3)
        mask.create_array(
            name = "0",
            shape=shape,
            chunks=zplane_chunks,
            shards=None,
            dtype="bool",
            dimension_names=('z', 'y', 'x'),
            compressors=[zarr.codecs.BloscCodec(cname='lz4', clevel=5)]
        )


def save_intensity_summary(results, channel):
    """compute a minimum and maximum (95% percentile) intensity value for the entire array
    This will be useful for normalization in e.g. nnUNet later"""

    #calculate minimum and 95% maximum values 
    minima = [float(low) for _, low, _ in results if np.isfinite(low)]
    percentiles = [float(high) for _, _, high in results if np.isfinite(high)]
    summary = {
        'local_min': min(minima) if minima else None,
        'local_95': max(percentiles) if percentiles else None,
    }

    #write out to a per-channel intensity.json file 
    path = Path(str(channel['zarr_output']) + '.intensity.json')
    with path.open('w', encoding='utf-8') as stream:
        json.dump(summary, stream, indent=2, allow_nan=False)
        stream.write('\n')
    log('Saved intensity statistics: ' + str(path))


def _write_entire_shard(source_path, target_path, zplane_start, shards):
    """copy over shards from the original array to the new (3D chunked) array,
    let zarr's logic sort out the details. Reads are aligned to the sharding of the new array
    to make sure each bit of the array is only read/written once"""

    #open source array
    source = zarr.open_group(source_path, mode='a', zarr_format=3)

    #open target array
    target = zarr.open_group(target_path, mode='a', zarr_format=3)

    #direct copy without creating a separate in-memory object
    target["0"][zplane_start:zplane_start+shards[0],:,:] = source["0"][zplane_start:zplane_start+shards[0],:,:]

    #debug
    #print(f"copied over data from zplanes {zplane_start, zplane_start+shards[0]}")

    return


def rechunk_channel(config, channel):
    """convert from the initial chunking with 1 zplane = 1 chunk 
    to a volume with 3D chunks, as specified in the config.json"""

    #open source array, calculate padded shape of the target array
    source_path = Path(channel['zarr_output']).resolve()
    source = zarr.open_group(str(source_path), mode='r')
    chunks = tuple(config['chunk_shape'])
    shards=(chunks[0],
        chunks[1] * math.ceil(source["0"].shape[1] / chunks[1]),
        chunks[2] * math.ceil(source["0"].shape[2] / chunks[2]))
    
    #sanity-check 
    if source["0"].chunks == chunks:
        log('%s: level 0 already has chunks %s' % (channel['name'], chunks))
        return
    log('%s: rechunking level 0 from %s to %s' %
        (channel['name'], source["0"].chunks, chunks))

    #create target array, same in almost all ways except that the chunking/sharding is handled as specified in the config.json
    target_path = Path(tempfile.mkdtemp(prefix=source_path.name + '.rechunk-', dir=source_path.parent))
    try:
        options = dict(
            shape=source["0"].shape, dtype=source["0"].dtype, 
            dimension_names=source["0"].metadata.dimension_names,
            compressors=source["0"].compressors, filters=source["0"].filters,
            serializer=source["0"].serializer, fill_value=source["0"].fill_value,
            attributes=dict(source["0"].attrs),
        )
       
        target = zarr.open_group(target_path, mode='w-', zarr_format=3)
        target.create_array("0", chunks=chunks, shards=shards, **options)
        
        #compute the steps that are needed to copy an entire shard at once (to minimize rewriting)
        zplane_chunk_starts = np.arange(0,source["0"].shape[0],chunks[0])

        #parallel copy
        tasks = [
            delayed(_write_entire_shard)(source_path, target_path, zplane_start, shards)
            for zplane_start in zplane_chunk_starts
        ]
        compute(*tasks)                

        #cleanup: remove source dir
        shutil.rmtree(source_path)
        #now, rename the target_dir to source_dir
        shutil.move(target_path, source_path)

    finally:
        log('%s: level 0 rechunked; temporary arrays removed' % channel['name'])
    


def _downsample_block(source_path, target_path, start, shards, source_shape):
    """Downsample a single block of the source array into the target array."""

    #open arrays 
    source = zarr.open_array(source_path, mode='r')
    target = zarr.open_array(target_path, mode='a')

    #downsample by this much in z/y/x
    factors = (1, 2, 2)

    #calculate shape for the downsampled array
    target_shape = tuple((n + f - 1) // f for n, f in zip(source_shape, factors))
    end_target = tuple(min(s + c, n) for s, c, n in zip(start, shards, target_shape))
    source_slice = tuple(slice(s * f, min(e * f, n))
                         for s, e, n, f in zip(start, end_target, source_shape, factors))
    
    #load data and sanity-check
    data = source[source_slice]
    if np.issubdtype(data.dtype, np.floating):
        value = -np.inf
    elif np.issubdtype(data.dtype, np.bool_):
        value = False
    else:
        value = np.iinfo(data.dtype).min

    # downsample, then insert into target array
    reduced = measure.block_reduce(data, block_size=factors, func=np.max, cval=value)
    target_slice = tuple(slice(s, e) for s, e in zip(start, end_target))
    target[target_slice] = reduced


def downsample_channel(config, channel):
    """Downsample the channel using Dask's parallel processing."""

    #open source array, gather chunking/sharding info
    group = zarr.open_group(channel["zarr_output"])
    chunks = tuple(config['chunk_shape'])
    shards = group["0"].shards
    log(f"{channel['name']}: stitching complete; building two max-downsampled levels")

    #create new level 
    for level in (1, 2):
        source = group[str(level - 1)]
        shape = (source.shape[0], (source.shape[1] + 1)//2, (source.shape[2] + 1)//2)
        target = group.create_array(
            str(level), 
            shape=shape, 
            chunks=chunks,
            shards=shards,
            dtype=source.dtype, 
            dimension_names=('z', 'y', 'x'),
            compressors=source.compressors)
        total_blocks = math.prod((n + c - 1) // c for n, c in zip(shape, shards))
        log(f"{channel['name']}: level {level}, shape={shape}, {total_blocks} blocks, {config['num_workers']} workers")
        starts = product(*(range(0, size, shard) for size, shard in zip(shape, shards)))
        source_path = os.path.join(channel['zarr_output'], str(level - 1))
        target_path = os.path.join(channel['zarr_output'], str(level))

        #gather arguments for parallel downsampling 
        args = []
        for start in starts:
            args.append((source_path, target_path, start, shards, source.shape))

        #downsample one shard at a time 
        downsample_tasks = [delayed(_downsample_block)(*arg) for arg in args]
        compute(*downsample_tasks)

        #add metadata for the newly downsampled array
        group.attrs['ome'] = level_metadata(config, channel, range(level + 1))

    log('Completed: ' + channel['zarr_output'])


def channel_layout(template, channel_number):
    """Reuse the C00 layout coordinates with this channel's image filenames."""
    if not re.fullmatch(r'C\d{2}', channel_number):
        raise ValueError('channel_number must be a channel ID such as C00 or C01')
    layout, count = re.subn(r'C03(?!\d)', channel_number, template)
    if count == 0:
        raise ValueError('Shared tile configuration must contain C00 references')
    return layout


def _stitch_single_plane(plane, channel, template, layout_dir, config, initial_run=False):
    """Stitches single z-plane, creates/applies mask, and stores in zarr array. Parallelizable. """
    # Build the per‑plane registered layout file
    layout, count = re.subn(r'Z\d{4}(?!\d)', 'Z%04d' % plane, template)
    if count == 0:
        raise ValueError('No Z#### references in tile configuration')
    working_config = Path(tempfile.mktemp(suffix='.registered'))
    working_config.write_text(layout)

    # Start headless Fiji in this worker
    scyjava.config.add_option('-Xmx' + config.get('java_heap', '8g'))
    ij = imagej.init(config['fiji_dir'], mode='headless')
    try:
        Stitcher = scyjava.jimport('Stitch_Image_Collection')
        stitcher = Stitcher()
        stitcher.rgbOrder = 'rgb'
        stitcher.alpha = 1.5
        stitcher.thresholdR = 0.95
        stitcher.thresholdDisplacementRelative = 150.0
        stitcher.thresholdDisplacementAbsolute = 350.0
        image = stitcher.work(
            working_config.as_posix(), False, config['compute_overlap'],
            'Linear Blending', 'Red, Green and Blue', False
            )
        if image is None:
            raise RuntimeError('Stitching failed at Z%04d' % plane)

        #convert from ImageJ ImagePlus to np array
        pixels = plane_pixels(image)          
                
        #Determine the correct z‑index for this plane
        z_index = plane - config['start_plane']
        
        #in the very beginning, run a truncated version because the zarrs still need to be set up
        if initial_run:
            print (f"initial run succeeded, dimensions are {pixels.shape}")
            return pixels

        #threshold if required 
        if channel["name"] == config["threshold_reference_channel"]:
            threshold = channel['threshold']
            pixels[pixels < threshold] = 0

            #debug 
            #print(f"trying to open first zarr, threshold: {threshold}")

            #write out to mask zarr
            mask_location = config["channels"][0]["zarr_output"]+ '.threshold-mask.zarr'
            mask = zarr.open_group(mask_location, mode='a')
            #pad pixels so that they fit 
            target_shape = mask['0'][z_index,:,:].shape
            pixels = np.pad(pixels,((0,target_shape[0]-pixels.shape[0]),(0,target_shape[1]-pixels.shape[1])))
            #write to zarr array
            mask['0'][z_index,:,:] = pixels.astype('bool')

        else:
            #if not reference chanel, open the masked location and import the mask from there 
            mask_location = config["channels"][0]["zarr_output"]+ '.threshold-mask.zarr'
            mask = zarr.open_group(mask_location, mode='r')
            mask_plane = mask['0'][z_index,:,:] 
            #pad pixels so that they fit 
            target_shape = mask['0'][z_index,:,:].shape
            pixels = np.pad(pixels,((0,target_shape[0]-pixels.shape[0]),(0,target_shape[1]-pixels.shape[1])))
            pixels = pixels * mask_plane 

        #write out to channel's zarr file 
        group = zarr.open_group(channel["zarr_output"], mode='a')
        group['0'][z_index,:,:] = pixels   # or group['0'][z_index] = pixels depending on layout
         
        #compute local z-plane minimum (above 0) and 95% percentile
        nonzero = pixels[pixels != 0]
        local_min = np.nanmin(nonzero) if nonzero.size else np.nan
        local_95 = np.nanpercentile(nonzero, 95) if nonzero.size else np.nan 

        #cleanup
        image.changes = False
        image.close()
        image.flush()
        
        #write out local min & 95%
        return plane, local_min, local_95
       
    finally:
        #more cleanup
        ij.dispose()
        working_config.unlink(missing_ok=True)


def stitch_channel(config, channel):
    """Per channel, creates 1 zarr folder (+mask) and runs stitching in parallel."""
    #start Fiji 
    log('%s: loading PyImageJ and starting headless Fiji (this can take a while)' % channel['name'])
    startup = time.perf_counter()
    scyjava.config.add_option('-Xmx' + config.get('java_heap', '8g'))
    ij = imagej.init(config['fiji_dir'], mode='headless')
    log('%s: Fiji ready after %.1fs' % (channel['name'], time.perf_counter() - startup))

    #gather tile config information 
    tile_config = Path(config['tile_config'])
    template = channel_layout(tile_config.read_text(), channel['channel_number'])
    layout_dir = tile_config.parent

    #get start and stop planes 
    planes = list(range(config['start_plane'], config['stop_plane']))

    #do one test stitching to get x&y dimensions, then create zarr 
    pixels = _stitch_single_plane(planes[0], channel, template, layout_dir, config, initial_run=True) 

    #create zarrs 
    create_groups(config, channel, pixels)
    
    #stitch planes in parallel 
    tasks = [
        delayed(_stitch_single_plane)(p, channel, template, layout_dir, config)
        for p in planes
    ]
    results = compute(*tasks)

    save_intensity_summary(results, channel)

    # Cleanup the main Fiji instance (workers have their own)
    ij.dispose()


if __name__ == '__main__':
    #parse commandline arguments, if any 
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('config_file', nargs='?', 
                        default='config_stitch.json',
                        help='Path to the JSON configuration file')
    args = parser.parse_args()

    # Load configuration
    config = load_config(args.config_file)
    
    #run the stitching and downsampling, reference channel first 
    if not config.get('validate_only'):
        #extract reference channel
        reference_name = config.get('threshold_reference_channel')
        
        #sort reference channel to be processed first
        config['channels'].sort(key=lambda c: c['name'] != reference_name)
    
        #process channels sequentially
        for current_channel in config["channels"]:
            log(f"Starting pipeline for channel: {current_channel}")
            stitch_channel(config, current_channel)
            
            rechunk_channel(config, current_channel)
            
            downsample_channel(config, current_channel)
            
            log(f"Finished pipeline for channel: {current_channel}")
    else:    
        #Optional: sanity-check the input 
        print('Configuration valid; %d channel(s). (Validation mode active)' % len(config['channels']))
