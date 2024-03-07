from histomicstk.cli.utils import CLIArgumentParser

if __name__ == '__main__':
    args = CLIArgumentParser().parse_args()
    if args.certainty == 'batchbald':
        from SuperpixelClassificationTorch import SuperpixelClassificationTorch

        superpixel_classification = SuperpixelClassificationTorch()
    else:
        from SuperpixelClassificationTensorflow import \
            SuperpixelClassificationTensorflow

        superpixel_classification = SuperpixelClassificationTensorflow()

    superpixel_classification.main(args)
