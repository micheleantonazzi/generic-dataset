#######################################################
# This stub file is automatically generated by stub-generator
# https://pypi.org/project/stub-generator/
########################################################

import numpy
import generic_dataset.utilities.save_load_methods
import generic_dataset.generic_sample
import generic_dataset.sample_generator
import generic_dataset.data_pipeline

def field_3_is_positive(sample) -> bool:
	...

class GeneratedSampleClassification(generic_dataset.generic_sample.GenericSample, metaclass=generic_dataset.sample_generator.MetaSample):
	def __init__(sample, label: int = 0):
		...
	def set_field_3(sample, value: int) -> GeneratedSampleClassification:
		"""
		Sets "field_3" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raises FieldHasIncorrectTypeException: if the given value has a wrong type
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before setting a new value
		:param value: the value to be assigned to field_3
		:type value: int
		:return: the GeneratedSampleClassification object
		:rtype: GeneratedSampleClassification
		"""
		...
	def get_field_3(sample) -> int:
		"""
		Return "field_3" value.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised. Terminate it before get the fields value
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before getting a new value
		:return: the value of field_3
		:rtype: int
		"""
		...
	def set_bgr_image(sample, value: numpy.ndarray) -> GeneratedSampleClassification:
		"""
		Sets "bgr_image" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raises FieldHasIncorrectTypeException: if the given value has a wrong type
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before setting a new value
		:param value: the value to be assigned to bgr_image
		:type value: ndarray
		:return: the GeneratedSampleClassification object
		:rtype: GeneratedSampleClassification
		"""
		...
	def get_bgr_image(sample) -> numpy.ndarray:
		"""
		Return "bgr_image" value.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised. Terminate it before get the fields value
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before getting a new value
		:return: the value of bgr_image
		:rtype: ndarray
		"""
		...
	def create_pipeline_for_bgr_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "bgr_image".
		The pipeline is correctly configured, the data to elaborate are "bgr_image"
		and the pipeline results is set to "bgr_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException: if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "bgr_image" and writes the result into "bgr_image"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_bgr_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of bgr_image. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_rgb_image(sample, value: numpy.ndarray) -> GeneratedSampleClassification:
		"""
		Sets "rgb_image" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raises FieldHasIncorrectTypeException: if the given value has a wrong type
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before setting a new value
		:param value: the value to be assigned to rgb_image
		:type value: ndarray
		:return: the GeneratedSampleClassification object
		:rtype: GeneratedSampleClassification
		"""
		...
	def get_rgb_image(sample) -> numpy.ndarray:
		"""
		Return "rgb_image" value.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised. Terminate it before get the fields value
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before getting a new value
		:return: the value of rgb_image
		:rtype: ndarray
		"""
		...
	def create_pipeline_for_rgb_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "rgb_image".
		The pipeline is correctly configured, the data to elaborate are "rgb_image"
		and the pipeline results is set to "rgb_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException: if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "rgb_image" and writes the result into "rgb_image"
		:rtype: DataPipeline
		"""
		...
	def get_pipeline_rgb_image(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Returns the pipeline of rgb_image. If there isn't an active pipeline, returns None.
		:return: the pipeline if it exists, None otherwise
		:rtype: Union[None, DataPipeline]
		"""
		...
	def set_label(sample, value: int) -> GeneratedSampleClassification:
		"""
		Sets "label" parameter.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised.
		:raises FieldHasIncorrectTypeException: if the given value has a wrong type
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before setting a new value
		:param value: the value to be assigned to label
		:type value: int
		:return: the GeneratedSampleClassification object
		:rtype: GeneratedSampleClassification
		"""
		...
	def get_label(sample) -> int:
		"""
		Return "label" value.
		If the field is an numpy.ndarray and it has an active pipeline, an exception is raised. Terminate it before get the fields value
		:raises AnotherActivePipelineException: if the field has an active pipeline, terminate it before getting a new value
		:return: the value of label
		:rtype: int
		"""
		...
	def create_pipeline_convert_rgb_to_bgr(sample) -> generic_dataset.data_pipeline.DataPipeline:
		"""
		Creates and returns a new pipeline to elaborate "rgb_image".
		The pipeline is correctly configured, the data to elaborate are "rgb_image"
		and the pipeline results is set to "bgr_image".
		If there is another active pipeline for this field, it raises an AnotherActivePipelineException.
		:raise AnotherActivePipelineException: if other pipeline of the fields are active
		:return: a new pipeline instance which elaborates "rgb_image" and writes the result into "bgr_image"
		:rtype: DataPipeline
		"""
		...
	def field_3_is_positive(sample) -> bool:
		...
	def save_field(sample, field_name: str, path: str, file_name: str) -> GeneratedSampleClassification:
		"""
		Saves the given field to disk, in the given path.
		:raise FieldDoesNotExistException: if field_name do not exist in this sample class
		:raise FieldIsNotDatasetPart: if the field is not a part of dataset
		:raise AnotherActivePipelineException if there is an active pipeline for this field
		:raise FileNotFoundError: if the path does not exist
		:param field_name: the name of the field to save
		:type field_name: str
		:param path: the path where save the field value
		:type path: str
		:param file_name: the name of the file in which to save the field. The file extension is automatically added by the save function, so don't include it in the name
		:type file_name: str
		:returns the sample instance
		"""
		...
	def load_field(sample, field_name: str, path: str, file_name: str) -> GeneratedSampleClassification:
		"""
		Loads the given field from disk, saved in the given path.
		The field value is not returned by this method but it is set to the sample class.
		To retrieve it use the correspondent get method.
		:raise FieldDoesNotExistException: if field_name do not exist in this sample class
		:raise FieldIsNotDatasetPart: if the field is not a part of dataset
		:raise AnotherActivePipelineException if there is an active pipeline for this field
		:raise FileNotFoundError: if the path does not exist
		:param field_name: the name of the field to save
		:type field_name: str
		:param path: the path where loading the field value
		:type path: str
		:param file_name: the name of the file in which the field si saved. The file extension is automatically added by the load function, so don't include it in the name
		:type file_name: str
		:return: the sample instance
		"""
		...
	def release_all_locks(sample) -> GeneratedSampleClassification:
		"""
		Releases all locks related to all fields of the sample instance
		:return: GenericSample instance
		"""
		...
	def acquire_all_locks(sample) -> GeneratedSampleClassification:
		"""
		Acquires all locks related to all fields of the sample instance
		:return: GenericSample instance
		"""
		...
	def __exit__(sample, exc_type, exc_value, exc_traceback):
		...
	def __enter__(sample) -> GeneratedSampleClassification:
		...
	...

