#!/usr/bin/env python
'''
This code is similar to girder_annotation/girder_large_image_annotation/models/annotation.py
The meaning is to validate the json annotation file without having to use girder or large_image
'''
import argparse
import json
import logging
import os
import sys
import jsonschema
from tqdm import tqdm

import copy

def extendSchema(base, add):
    extend = copy.deepcopy(base)
    for key in add:
        if key == 'required' and 'required' in base:
            extend[key] = sorted(set(extend[key]) | set(add[key]))
        elif key != 'properties' and 'properties' in base:
            extend[key] = add[key]
    if 'properties' in add:
        extend['properties'].update(add['properties'])
    return extend


colorSchema = {
    'type': 'string',
    # We accept colors of the form
    #   #rrggbb                 six digit RRGGBB hex
    #   #rgb                    three digit RGB hex
    #   #rrggbbaa               eight digit RRGGBBAA hex
    #   #rgba                   four digit RGBA hex
    #   rgb(255, 255, 255)      rgb decimal triplet
    #   rgba(255, 255, 255, 1)  rgba quad with RGB in the range [0-255] and
    #                           alpha [0-1]
    'pattern': r'^(#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})|'
               r'rgb\(\d+,\s*\d+,\s*\d+\)|'
               r'rgba\(\d+,\s*\d+,\s*\d+,\s*(\d?\.|)\d+\))$',
}

transformArray = {
    'type': 'array',
    'items': {
        'type': 'array',
        'minItems': 2,
        'maxItems': 2,
    },
    'minItems': 2,
    'maxItems': 2,
    'description': 'A 2D matrix representing the transform of an '
                    'image overlay.',
}


colorRangeSchema = {
    'type': 'array',
    'items': colorSchema,
    'description': 'A list of colors',
}

rangeValueSchema = {
    'type': 'array',
    'items': {'type': 'number'},
    'description': 'A weakly monotonic list of range values',
}

userSchema = {
    'type': 'object',
    'additionalProperties': True,
}

labelSchema = {
    'type': 'object',
    'properties': {
        'value': {'type': 'string'},
        'visibility': {
            'type': 'string',
            # TODO: change to True, False, None?
            'enum': ['hidden', 'always', 'onhover'],
        },
        'fontSize': {
            'type': 'number',
            'exclusiveMinimum': 0,
        },
        'color': colorSchema,
    },
    'required': ['value'],
    'additionalProperties': False,
}

groupSchema = {'type': 'string'}

baseElementSchema = {
    'type': 'object',
    'properties': {
        'id': {
            'type': 'string',
            'pattern': '^[0-9a-f]{24}$',
        },
        'type': {'type': 'string'},
        # schema free field for users to extend annotations
        'user': userSchema,
        'label': labelSchema,
        'group': groupSchema,
    },
    'required': ['type'],
    'additionalProperties': True,
}
baseShapeSchema = extendSchema(baseElementSchema, {
    'properties': {
        'lineColor': colorSchema,
        'lineWidth': {
            'type': 'number',
            'minimum': 0,
        },
    },
})


pixelmapCategorySchema = {
    'type': 'object',
    'properties': {
        'fillColor': colorSchema,
        'strokeColor': colorSchema,
        'label': {
            'type': 'string',
            'description': 'A string representing the semantic '
                           'meaning of regions of the map with '
                           'the corresponding color.',
        },
        'description': {
            'type': 'string',
            'description': 'A more detailed explanation of the '
                           'meaining of this category.',
        },
    },
    'required': ['fillColor'],
    'additionalProperties': False,
}

_annotationSchema = {
    'type': 'object',
    'properties': {
        'value': colorSchema,
        'id': colorSchema,
        'label': {
            'type': 'string',
            'description': 'A string representing the semantic '
                           'meaning of regions of the map with '
                           'the corresponding color.',
        },
        'description': {
            'type': 'string',
            'description': 'A more detailed explanation of the '
                           'meaining of this category.',
        },
    },
    'required': ['fillColor'],
    'additionalProperties': False,
}


overlaySchema = extendSchema(baseElementSchema, {
    'properties': {
        'type': {
            'type': 'string',
            'enum': ['image'],
        },
        'girderId': {
            'type': 'string',
            'pattern': '^[0-9a-f]{24}$',
            'description': 'Girder item ID containing the image to '
                            'overlay.',
        },
        'opacity': {
            'type': 'number',
            'minimum': 0,
            'maximum': 1,
            'description': 'Default opacity for this image overlay. Must '
                            'be between 0 and 1. Defaults to 1.',
        },
        'hasAlpha': {
            'type': 'boolean',
            'description':
                'If true, the image is treated assuming it has an alpha '
                'channel.',
        },
        'transform': {
            'type': 'object',
            'description': 'Specification for an affine transform of the '
                            'image overlay. Includes a 2D transform matrix, '
                            'an X offset and a Y offset.',
            'properties': {
                'xoffset': {
                    'type': 'number',
                },
                'yoffset': {
                    'type': 'number',
                },
                'matrix': transformArray,
            },
        },
    },
    'required': ['girderId', 'type'],
    'additionalProperties': False,
    'description': 'An image overlay on top of the base resource.',
})


pixelmapSchema = extendSchema(overlaySchema, {
    'properties': {
        'type': {
            'type': 'string',
            'enum': ['pixelmap'],
        },
        'values': {
            'type': 'array',
            'items': {'type': 'integer'},
            'description': 'An array where the indices '
                           'correspond to pixel values in the '
                           'pixel map image and the values are '
                           'used to look up the appropriate '
                           'color in the categories property.',
        },
        'categories': {
            'type': 'array',
            'items': pixelmapCategorySchema,
            'description': 'An array used to map between the '
                           'values array and color values. '
                           'Can also contain semantic '
                           'information for color values.',
        },
        'boundaries': {
            'type': 'boolean',
            'description': 'True if the pixelmap doubles pixel '
                           'values such that even values are the '
                           'fill and odd values the are stroke '
                           'of each superpixel. If true, the '
                           'length of the values array should be '
                           'half of the maximum value in the '
                           'pixelmap.',

        },
    },
    'required': ['values', 'categories', 'boundaries'],
    'additionalProperties': False,
    'description': 'A tiled pixelmap to overlay onto a base resource.',
})

bboxSchema = extendSchema(overlaySchema, {
    'properties': {
        'type': {
            'type': 'string',
            'enum': ['bboxmap'],
        },
        'categories': {
            'type': 'array',
            'items': pixelmapCategorySchema,
            'description': 'An array used to map between the '
                           'values array and color values. '
                           'Can also contain semantic '
                           'information for color values.',
        },
        'annotations': {
            'type': 'array',
            'description': 'Value, id, and bounding box for each annotation',
                'items': {
                'type': 'object',
                'additionalProperties': False,
                'properties': {
                    'value': {
                        'type': 'integer',
                    },
                    'id': {
                        'type': 'integer',
                    },
                    'bbox': {
                        'type': 'array',
                        'items': {'type': 'number'},
                        'minItems': 4,
                        'maxItems': 4,
                        'description': 'Bounding box in the form '
                                       '[left, top, right, bottom].',
                    },
                }
            }
        },
        'boundaries': {
            'type': 'boolean',
            'description': 'True if the pixelmap doubles pixel '
                           'values such that even values are the '
                           'fill and odd values the are stroke '
                           'of each superpixel. If true, the '
                           'length of the values array should be '
                           'half of the maximum value in the '
                           'pixelmap.',

        },
    },
    'required': ['categories', 'boundaries', 'annotations'],
    'additionalProperties': True,
    'description': 'A tiled pixelmap to overlay onto a base resource.',
})

annotationElementSchema = {
    # Shape subtypes are mutually exclusive, so for efficiency, don't use
    # 'oneOf'
    'anyOf': [
        pixelmapSchema,
        bboxSchema,
    ],
}


class AnnotationSchema:
    annotationSchema = {
        '$schema': 'http://json-schema.org/schema#',
        'type': 'object',
        'properties': {
            'name': {
                'type': 'string',
                # TODO: Disallow empty?
                'minLength': 1,
            },
            'description': {'type': 'string'},
            'display': {
                'type': 'object',
                'properties': {
                    'visible': {
                        'type': ['boolean', 'string'],
                        'enum': ['new', True, False],
                        'description': 'This advises viewers on when the '
                        'annotation should be shown.  If "new" (the default), '
                        'show the annotation when it is first added to the '
                        "system.  If false, don't show the annotation by "
                        'default.  If true, show the annotation when the item '
                        'is displayed.',
                    },
                },
            },
            'attributes': {
                'type': 'object',
                'additionalProperties': True,
                'title': 'Image Attributes',
                'description': 'Subjective things that apply to the entire '
                               'image.',
            },
            'elements': {
                'type': 'array',
                'items': annotationElementSchema,
                # We want to ensure unique element IDs, if they are set.  If
                # they are not set, we assign them from Mongo.
                'title': 'Image Markup',
                'description': 'Subjective things that apply to a '
                               'spatial region.',
            },
        },
        'additionalProperties': False,
    }



    coordSchema = {
        'type': 'array',
        # TODO: validate that z==0 for now
        'items': {
            'type': 'number',
        },
        'minItems': 3,
        'maxItems': 3,
        'name': 'Coordinate',
        # TODO: define origin for 3D images
        'description': 'An X, Y, Z coordinate tuple, in base layer pixel '
                       'coordinates, where the origin is the upper-left.',
    }
    coordValueSchema = {
        'type': 'array',
        'items': {
            'type': 'number',
        },
        'minItems': 4,
        'maxItems': 4,
        'name': 'CoordinateWithValue',
        'description': 'An X, Y, Z, value coordinate tuple, in base layer '
                       'pixel coordinates, where the origin is the upper-left.',
    }

    colorSchema = {
        'type': 'string',
        # We accept colors of the form
        #   #rrggbb                 six digit RRGGBB hex
        #   #rgb                    three digit RGB hex
        #   #rrggbbaa               eight digit RRGGBBAA hex
        #   #rgba                   four digit RGBA hex
        #   rgb(255, 255, 255)      rgb decimal triplet
        #   rgba(255, 255, 255, 1)  rgba quad with RGB in the range [0-255] and
        #                           alpha [0-1]
        # TODO: make rgb and rgba spec validate that rgb is [0-255] and a is
        # [0-1], rather than just checking if they are digits and such.
        'pattern': r'^(#([0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})|'
                   r'rgb\(\d+,\s*\d+,\s*\d+\)|'
                   r'rgba\(\d+,\s*\d+,\s*\d+,\s*(\d?\.|)\d+\))$',
    }

    colorRangeSchema = {
        'type': 'array',
        'items': colorSchema,
        'description': 'A list of colors',
    }

    rangeValueSchema = {
        'type': 'array',
        'items': {'type': 'number'},
        'description': 'A weakly monotonic list of range values',
    }

    userSchema = {
        'type': 'object',
        'additionalProperties': True,
    }

    labelSchema = {
        'type': 'object',
        'properties': {
            'value': {'type': 'string'},
            'visibility': {
                'type': 'string',
                # TODO: change to True, False, None?
                'enum': ['hidden', 'always', 'onhover'],
            },
            'fontSize': {
                'type': 'number',
                'exclusiveMinimum': 0,
            },
            'color': colorSchema,
        },
        'required': ['value'],
        'additionalProperties': False,
    }

    groupSchema = {'type': 'string'}

    baseElementSchema = {
        'type': 'object',
        'properties': {
            'id': {
                'type': 'string',
                'pattern': '^[0-9a-f]{24}$',
            },
            'type': {'type': 'string'},
            # schema free field for users to extend annotations
            'user': userSchema,
            'label': labelSchema,
            'group': groupSchema,
        },
        'required': ['type'],
        'additionalProperties': True,
    }
    baseShapeSchema = extendSchema(baseElementSchema, {
        'properties': {
            'lineColor': colorSchema,
            'lineWidth': {
                'type': 'number',
                'minimum': 0,
            },
        },
    })

    pointShapeSchema = extendSchema(baseShapeSchema, {
        'properties': {
            'type': {
                'type': 'string',
                'enum': ['point'],
            },
            'center': coordSchema,
            'fillColor': colorSchema,
        },
        'required': ['type', 'center'],
        'additionalProperties': False,
    })

    arrowShapeSchema = extendSchema(baseShapeSchema, {
        'properties': {
            'type': {
                'type': 'string',
                'enum': ['arrow'],
            },
            'points': {
                'type': 'array',
                'items': coordSchema,
                'minItems': 2,
                'maxItems': 2,
            },
            'fillColor': colorSchema,
        },
        'description': 'The first point is the head of the arrow',
        'required': ['type', 'points'],
        'additionalProperties': False,
    })

    circleShapeSchema = extendSchema(baseShapeSchema, {
        'properties': {
            'type': {
                'type': 'string',
                'enum': ['circle'],
            },
            'center': coordSchema,
            'radius': {
                'type': 'number',
                'minimum': 0,
            },
            'fillColor': colorSchema,
        },
        'required': ['type', 'center', 'radius'],
        'additionalProperties': False,
    })

    polylineShapeSchema = extendSchema(baseShapeSchema, {
        'properties': {
            'type': {
                'type': 'string',
                'enum': ['polyline'],
            },
            'points': {
                'type': 'array',
                'items': coordSchema,
                'minItems': 2,
            },
            'fillColor': colorSchema,
            'closed': {
                'type': 'boolean',
                'description': 'polyline is open if closed flag is '
                               'not specified',
            },
            'holes': {
                'type': 'array',
                'description':
                    'If closed is true, this is a list of polylines that are '
                    'treated as holes in the base polygon. These should not '
                    'cross each other and should be contained within the base '
                    'polygon.',
                'items': {
                    'type': 'array',
                    'items': coordSchema,
                    'minItems': 3,
                },
            },
        },
        'required': ['type', 'points'],
        'additionalProperties': False,
    })


def validate_annotation(annotation_dict):
    validator = jsonschema.Draft6Validator(AnnotationSchema.annotationSchema)
    validatorElement = jsonschema.Draft6Validator(AnnotationSchema.baseElementSchema)

    validator.validate(annotation_dict)
    for element in tqdm(annotation_dict['elements']):
        validatorElement.validate(element)

def validate_json_file(json_dst):
    with open(json_dst, 'r') as f:
        data = json.load(f)
        validate_annotation(data)
        # num_elem = len(data['elements'][0]['annotations'])
        # if num_elem % 4 != 0:
        #     raise ValueError(f"Number of elements ({num_elem}) is not a multiple of 4")
        # num_values = len(data['elements'][0]['annotations'])
        # if int(num_elem / 4) != num_values:
        #     raise ValueError(f"Number of elements ({num_elem / 4}) does not match values ({num_values})")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Validate a json annotation file')
    parser.add_argument('--input', default=os.path.join("out", "superpixel.anot"), type=str,
                        help='Name of input json file with a pixelmap annotation"')
    args = parser.parse_args()
    # Call the function with the filenames
    logging.basicConfig(stream=sys.stdout, level=logging.INFO)

    if not os.path.exists(args.input):
        logging.error(f"Annotation path {args.input} does not exist")
        sys.exit(1)

    validate_json_file(args.input)
    logging.info("Done validating annotation ['%s']", args.input)
