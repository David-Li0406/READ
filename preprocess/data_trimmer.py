'''
Short script to produce debug versions of the training data to aid in development
'''
import json
import os

def create_debug_dataset(config):
    # Filenames and vars
    orig_data_dir = os.path.join(config.data_dir, 'docred')
    debug_data_dir = os.path.join(config.data_dir, 'DocRED_debug')

    if not os.path.exists(debug_data_dir):
        os.mkdir(debug_data_dir)

    fname_and_slice = [
        ('dev.json', 100),
        ('test.json', 100),
        ('train_distant.json', 1000),
        ('train_annotated.json', 100),
    ]

    for out_fname_debug_data, size_of_slice in fname_and_slice:
        # Output file
        out_file = os.path.join(debug_data_dir,out_fname_debug_data)

        # Load full data
        full_data = json.load(open(os.path.join(orig_data_dir,out_fname_debug_data)))

        # Create slice
        debug_slice = full_data[:size_of_slice]

        # Write new file
        open(out_file,'w').write("[%s]" % ",\n ".join(json.dumps(e) for e in debug_slice))

        print('Created file:', out_fname_debug_data)
