from data.converter import Converter
from data.inputs import Inputs
from helpers.configuration import Configuration


config = Configuration()
inputs = Inputs(config)
converter = Converter(config)

training_inputs, validation_inputs, testing_inputs = inputs.read_files()

print('Converting training_set...')
converter.convert_to_tf_records(training_inputs, 'training_set')
print('Converting validation_set...')
converter.convert_to_tf_records(validation_inputs, 'validation_set')
print('Converting testing_set...')
converter.convert_to_tf_records(testing_inputs, 'testing_set')