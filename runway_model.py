# =========================================================================

# Import the Runway SDK. Please install it first with
# `pip install runway-python`.
import runway
from runway.data_types import number, text, image, category
from example_model import ExampleModel

# Setup the model, initialize weights, set the configs of the model, etc.
# Every model will have a different set of configurations and requirements.
# Check https://docs.runwayapp.ai/#/python-sdk to see a complete list of
# supported configs. The setup function should return the model ready to be
# used.
setup_options = {
    'truncation': number(min=1, max=10, step=1, default=5, description='Example input.'),
    'seed': number(min=0, max=1000000, description='A seed used to initialize the model.')
}
@runway.setup(options=setup_options)
def setup(opts):
    msg = '[SETUP] Ran with options: seed = {}, truncation = {}'
    print(msg.format(opts['seed'], opts['truncation']))
    model = ExampleModel(opts)
    return model

inputs = {
    # 'file': file(extension=".zip"),
    'image': image(),
    'model': category(choices=["none", "random", "color", "bit/m-r101x1", "vgg16"], default="color", description='Cluster model.'),
    'slices': number(min=5, max=30, step=5, default=10, description='Number of slices.'),
    'vgg_depth': number(min=1, max=8, step=1, default=7, description='VGG Feature Depth'),
}

# Every model needs to have at least one command. Every command allows to send
# inputs and process outputs. To see a complete list of supported inputs and
# outputs data types: https://sdk.runwayml.com/en/latest/data_types.html
@runway.command(name='generate',
                inputs=inputs,
                outputs={ 'image': image(width=512, height=512), 'info': text("hello") },
                description='Generates a red square when the input text input is "red".')
def generate(model, args):
    print('[GENERATE] Ran with image "{}"'.format(args['image']))
    return {
        'image': arg['image'], 'info': "hello world"
    }
    # Generate a PIL or Numpy image based on the input caption, and return it
    # output_image = model.run_on_input(args['image'], args['slices'], args['model'], args['vgg_depth'])
    # return {
    #     'image': output_image['image'], 'info': output_image['info']
    # }

if __name__ == '__main__':
    # run the model server using the default network interface and ports,
    # displayed here for convenience
    runway.run(host='0.0.0.0', port=9000)

## Now that the model is running, open a new terminal and give it a command to
## generate an image. It will respond with a base64 encoded URI
# curl \
#   -H "content-type: application/json" \
#   -d '{ "caption": "red" }' \
#   localhost:8000/generate
