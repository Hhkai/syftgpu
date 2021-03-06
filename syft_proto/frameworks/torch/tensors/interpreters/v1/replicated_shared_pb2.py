# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: syft_proto/frameworks/torch/tensors/interpreters/v1/replicated_shared.proto
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from google.protobuf import reflection as _reflection
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from syft_proto.generic.pointers.v1 import pointer_tensor_pb2 as syft__proto_dot_generic_dot_pointers_dot_v1_dot_pointer__tensor__pb2
from syft_proto.types.syft.v1 import id_pb2 as syft__proto_dot_types_dot_syft_dot_v1_dot_id__pb2


DESCRIPTOR = _descriptor.FileDescriptor(
  name='syft_proto/frameworks/torch/tensors/interpreters/v1/replicated_shared.proto',
  package='syft_proto.frameworks.torch.tensors.interpreters.v1',
  syntax='proto3',
  serialized_options=b'\n@org.openmined.syftproto.frameworks.torch.tensors.interpreters.v1',
  create_key=_descriptor._internal_create_key,
  serialized_pb=b'\nKsyft_proto/frameworks/torch/tensors/interpreters/v1/replicated_shared.proto\x12\x33syft_proto.frameworks.torch.tensors.interpreters.v1\x1a\x33syft_proto/generic/pointers/v1/pointer_tensor.proto\x1a!syft_proto/types/syft/v1/id.proto\"\xac\x02\n\x17ReplicatedSharingTensor\x12,\n\x02id\x18\x01 \x01(\x0b\x32\x1c.syft_proto.types.syft.v1.IdR\x02id\x12\x1b\n\x08ring_int\x18\x02 \x01(\x03H\x00R\x07ringInt\x12\x1b\n\x08ring_str\x18\x03 \x01(\tH\x00R\x07ringStr\x12\x14\n\x05\x64type\x18\x04 \x01(\tR\x05\x64type\x12?\n\x0clocation_ids\x18\x05 \x03(\x0b\x32\x1c.syft_proto.types.syft.v1.IdR\x0blocationIds\x12\x45\n\x06shares\x18\x06 \x03(\x0b\x32-.syft_proto.generic.pointers.v1.PointerTensorR\x06sharesB\x0b\n\tring_sizeBB\n@org.openmined.syftproto.frameworks.torch.tensors.interpreters.v1b\x06proto3'
  ,
  dependencies=[syft__proto_dot_generic_dot_pointers_dot_v1_dot_pointer__tensor__pb2.DESCRIPTOR,syft__proto_dot_types_dot_syft_dot_v1_dot_id__pb2.DESCRIPTOR,])




_REPLICATEDSHARINGTENSOR = _descriptor.Descriptor(
  name='ReplicatedSharingTensor',
  full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor',
  filename=None,
  file=DESCRIPTOR,
  containing_type=None,
  create_key=_descriptor._internal_create_key,
  fields=[
    _descriptor.FieldDescriptor(
      name='id', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.id', index=0,
      number=1, type=11, cpp_type=10, label=1,
      has_default_value=False, default_value=None,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='id', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ring_int', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.ring_int', index=1,
      number=2, type=3, cpp_type=2, label=1,
      has_default_value=False, default_value=0,
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='ringInt', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='ring_str', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.ring_str', index=2,
      number=3, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='ringStr', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='dtype', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.dtype', index=3,
      number=4, type=9, cpp_type=9, label=1,
      has_default_value=False, default_value=b"".decode('utf-8'),
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='dtype', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='location_ids', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.location_ids', index=4,
      number=5, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='locationIds', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
    _descriptor.FieldDescriptor(
      name='shares', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.shares', index=5,
      number=6, type=11, cpp_type=10, label=3,
      has_default_value=False, default_value=[],
      message_type=None, enum_type=None, containing_type=None,
      is_extension=False, extension_scope=None,
      serialized_options=None, json_name='shares', file=DESCRIPTOR,  create_key=_descriptor._internal_create_key),
  ],
  extensions=[
  ],
  nested_types=[],
  enum_types=[
  ],
  serialized_options=None,
  is_extendable=False,
  syntax='proto3',
  extension_ranges=[],
  oneofs=[
    _descriptor.OneofDescriptor(
      name='ring_size', full_name='syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor.ring_size',
      index=0, containing_type=None,
      create_key=_descriptor._internal_create_key,
    fields=[]),
  ],
  serialized_start=221,
  serialized_end=521,
)

_REPLICATEDSHARINGTENSOR.fields_by_name['id'].message_type = syft__proto_dot_types_dot_syft_dot_v1_dot_id__pb2._ID
_REPLICATEDSHARINGTENSOR.fields_by_name['location_ids'].message_type = syft__proto_dot_types_dot_syft_dot_v1_dot_id__pb2._ID
_REPLICATEDSHARINGTENSOR.fields_by_name['shares'].message_type = syft__proto_dot_generic_dot_pointers_dot_v1_dot_pointer__tensor__pb2._POINTERTENSOR
_REPLICATEDSHARINGTENSOR.oneofs_by_name['ring_size'].fields.append(
  _REPLICATEDSHARINGTENSOR.fields_by_name['ring_int'])
_REPLICATEDSHARINGTENSOR.fields_by_name['ring_int'].containing_oneof = _REPLICATEDSHARINGTENSOR.oneofs_by_name['ring_size']
_REPLICATEDSHARINGTENSOR.oneofs_by_name['ring_size'].fields.append(
  _REPLICATEDSHARINGTENSOR.fields_by_name['ring_str'])
_REPLICATEDSHARINGTENSOR.fields_by_name['ring_str'].containing_oneof = _REPLICATEDSHARINGTENSOR.oneofs_by_name['ring_size']
DESCRIPTOR.message_types_by_name['ReplicatedSharingTensor'] = _REPLICATEDSHARINGTENSOR
_sym_db.RegisterFileDescriptor(DESCRIPTOR)

ReplicatedSharingTensor = _reflection.GeneratedProtocolMessageType('ReplicatedSharingTensor', (_message.Message,), {
  'DESCRIPTOR' : _REPLICATEDSHARINGTENSOR,
  '__module__' : 'syft_proto.frameworks.torch.tensors.interpreters.v1.replicated_shared_pb2'
  # @@protoc_insertion_point(class_scope:syft_proto.frameworks.torch.tensors.interpreters.v1.ReplicatedSharingTensor)
  })
_sym_db.RegisterMessage(ReplicatedSharingTensor)


DESCRIPTOR._options = None
# @@protoc_insertion_point(module_scope)
