from histomicstk.cli.utils import CLIArgumentParser
from SuperpixelClassificationBase import SuperpixelClassificationBase

if __name__ == '__main__':
    args = CLIArgumentParser().parse_args()
    # Use tensorflow unless the dependency requires torch
    superpixel_classification: SuperpixelClassificationBase
    if args.certainty == 'batchbald' or args.feature == 'vector':
        from SuperpixelClassificationTorch import SuperpixelClassificationTorch

        superpixel_classification = SuperpixelClassificationTorch()
    else:
        from SuperpixelClassificationTensorflow import \
            SuperpixelClassificationTensorflow

        superpixel_classification = SuperpixelClassificationTensorflow()

    superpixel_classification.main(args)
