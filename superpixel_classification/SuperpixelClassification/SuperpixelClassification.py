from histomicstk.cli.utils import CLIArgumentParser

if __name__ == '__main__':
    from SuperpixelClassificationTensorflow import \
        SuperpixelClassificationTensorflow

    superpixel_classification = SuperpixelClassificationTensorflow()
    superpixel_classification.main(CLIArgumentParser().parse_args())
