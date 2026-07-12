from rfdetr import RFDETRSmall


def main() -> None:
    model = RFDETRSmall(
        pretrain_weights="/home/tom/Downloads/checkpoint_best_regular (12.7).pth",
        resolution=640,
        num_classes=2,
    )
    model.export(
        output_dir="/home/tom/Documents/Projeler/DogFight/object_detection",
        output_file_name="inference_model.onnx",
    )
    print("ONNX export tamamlandi: object_detection/inference_model.onnx")


if __name__ == "__main__":
    main()