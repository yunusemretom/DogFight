from rfdetr import RFDETRSmall


def main() -> None:
    model = RFDETRSmall(pretrain_weights="/home/tom/Downloads/checkpoint_best_regular_2.pth")
    model.export(
        output_dir="/home/tom/Documents/Dog_Fight/object_detection",
        output_file_name="inference_model.onnx",
    )
    print("ONNX export tamamlandi: object_detection/inference_model.onnx")


if __name__ == "__main__":
    main()