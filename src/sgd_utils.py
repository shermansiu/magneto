# Copyright 2021 Google Research.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utils for processing schema-guided dialogue data."""

import collections
import copy
import json
import pathlib
import re
import typing as tp

import logging

DialoguesDict = tp.Dict[str, tp.Any]
Schema = tp.Dict[str, tp.Any]
Schemas = tp.List[Schema]


def load_schemas_to_dict(data_dir: str, subdir: str,
                         output_dict: tp.Dict[str, Schemas]) -> None:
    """Loads a schema json from the given subdir into a provided dict."""
    schema_file_path = pathlib.Path(data_dir).joinpath(f"{subdir}/schema.json")
    with schema_file_path.open() as f:
        output_dict[subdir] = json.load(f)
        logging.info('Loaded schema file %s', schema_file_path)


def load_dialogues_to_dict(data_dir: str, subdir: str,
                           output_dict: tp.Dict[str, DialoguesDict]) -> None:
    """Loads dialogue jsons from the given subdir into a provided dict."""
    if subdir not in output_dict:
        output_dict[subdir] = {}
    dialogue_files = pathlib.Path(data_dir).joinpath(subdir).glob('dialogues*.json')
    for dialogue_file in dialogue_files:
        dialogue_filename = dialogue_file.name
        with dialogue_filename.open() as f:
            output_dict[subdir][dialogue_filename] = json.load(f)

        logging.info('Loaded dialogue file %s', dialogue_file)


def load_dataset(
    data_dir: str,
    subdirs: tp.List[str]
) -> tp.Tuple[tp.Dict[str, Schemas], tp.Dict[str, DialoguesDict]]:
    """Loads schemas and dialogues into dicts keyed by subdir."""
    subdir_to_schema = {}
    subdir_to_dialogues = collections.defaultdict(dict)
    for subdir in subdirs:
        load_schemas_to_dict(data_dir, subdir, subdir_to_schema)
        load_dialogues_to_dict(data_dir, subdir, subdir_to_dialogues)

    return subdir_to_schema, subdir_to_dialogues


def dedupe_and_unnest_schemas(
    subdir_to_schema: tp.Dict[str, Schemas]) -> tp.Dict[str, Schema]:
    """Deduplicates schemas and tags with the subdirs they come from.

    Args:
        subdir_to_schema: A dict mapping subdir to a list of schema dicts.

    Returns:
        A dict mapping each service name to its original schema, with an additional
        metadata field for which subdirs the schema is present in.
    """
    deduped_schemas = {}
    for subdir, schemas in subdir_to_schema.items():
        for schema in schemas:
            service_name = schema['service_name']

            # Add new schemas to dict
            if service_name not in deduped_schemas:
                schema_copy = copy.deepcopy(schema)
                schema_copy['subdirs'] = []
                deduped_schemas[service_name] = schema_copy

            # Update subdirs list
            deduped_schemas[service_name]['subdirs'].append(subdir)

    return deduped_schemas


def write_dialogue_dir(data_dir: str, subdir: str,
                       output_dict: tp.Dict[str, DialoguesDict]) -> None:
    """Writes dialogues from json object into files."""
    destination_dir = pathlib.Path(data_dir).joinpath(subdir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    for dialogue_filename in output_dict[subdir]:
        dialogue_file = destination_dir.joinpath(dialogue_filename)
        with dialogue_file.open("w") as output_dialogue_file:
          json.dump(
              output_dict[subdir][dialogue_filename],
              output_dialogue_file,
              indent=2,
              separators=(',', ': ')
          )
          logging.info('Wrote %s', dialogue_file)


def write_schema_dir(data_dir: str, subdir: str,
                     output_dict: tp.Dict[str, Schema]) -> None:
    """Writes schemas from json object into files."""
    destination_dir = pathlib.Path(data_dir).joinpath(subdir)
    destination_dir.mkdir(parents=True, exist_ok=True)
    schema_file = destination_dir.joinpath('schema.json')
    with schema_file.open('w') as output_schema_file:
      json.dump(
          output_dict[subdir],
          output_schema_file,
          indent=2,
          separators=(',', ': ')
      )
      logging.info('Wrote %s', schema_file)


def space_camel_case(s: str) -> tp.Optional[str]:
  """Returns a camel case string with spaces in between words."""
  if not s:
    return None

  words = re.findall(r'[A-Z](?:[a-z]+|[A-Z]*(?=[A-Z]|$))', s)
  return ' '.join(words)


def space_snake_case(s: str) -> tp.Optional[str]:
  """Returns a snake case string with spaces in between words."""
  return s.replace('_', ' ') if s else None


def nullsafe_str_join(l: tp.Optional[tp.List[str]], delimiter: str) -> tp.Optional[str]:
  """Performs string join on a list, and returns None for empty list or None."""
  return delimiter.join(l) if l else None
