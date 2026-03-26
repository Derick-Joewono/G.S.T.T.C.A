from predict import run_inference

if __name__ == "__main__":

    t1_image = "data/test/T1/sample.npy"
    t2_image = "data/test/T2/sample.npy"

    output_map = "./outputs/change_map.png"

    run_inference(t1_image, t2_image, output_map)


