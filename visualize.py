import os
import cv2


class YoloVisualizer:
    MODE_TRAIN = 0
    MODE_VAL = 1

    def __init__(self, dataset_folder, classes_file=None):
        self.dataset_folder = dataset_folder

        # Use provided classes_file or default to dataset_folder/classes.txt
        if classes_file is None:
            classes_file = os.path.join(dataset_folder, "classes.txt")

        if not os.path.exists(classes_file):
            raise FileNotFoundError(f"classes.txt not found at {classes_file}")

        with open(classes_file, "r") as f:
            self.classes = {i: c for i, c in enumerate(f.read().splitlines())}

        self.set_mode(YoloVisualizer.MODE_TRAIN)

    def set_mode(self, mode=MODE_TRAIN):
        if mode == self.MODE_TRAIN:
            self.images_folder = os.path.join(self.dataset_folder, "images")
            self.labels_folder = os.path.join(self.dataset_folder, "labels")
        else:
            self.images_folder = os.path.join(self.dataset_folder, "val1", "images")
            self.labels_folder = os.path.join(self.dataset_folder, "val1", "labels")

        if not os.path.exists(self.images_folder) or not os.path.exists(self.labels_folder):
            raise FileNotFoundError(f"Images or labels folder not found:\n{self.images_folder}\n{self.labels_folder}")

        self.image_names = sorted(os.listdir(self.images_folder))
        self.label_names = sorted(os.listdir(self.labels_folder))
        self.num_images = len(self.image_names)

        if self.num_images != len(self.label_names):
            raise ValueError("Mismatch between number of images and labels")
        if self.num_images == 0:
            raise ValueError("No images found")

        self.frame_index = 0

    def next_frame(self):
        self.frame_index = (self.frame_index + 1) % self.num_images

    def previous_frame(self):
        self.frame_index = (self.frame_index - 1) % self.num_images

    def seek_frame(self, idx):
        image_file = os.path.join(self.images_folder, self.image_names[idx])
        label_file = os.path.join(self.labels_folder, self.label_names[idx])

        if not os.path.exists(image_file) or not os.path.exists(label_file):
            return None

        with open(label_file, "r") as f:
            lines = f.read().splitlines()

        image = cv2.imread(image_file)
        for line in lines:
            class_index, x, y, w, h = map(float, line.split())
            cx = int(x * image.shape[1])
            cy = int(y * image.shape[0])
            w = int(w * image.shape[1])
            h = int(h * image.shape[0])
            x = cx - w // 2
            y = cy - h // 2
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(image, self.classes[int(class_index)], (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        return image

    def run(self):
        while True:
            frame = self.seek_frame(self.frame_index)
            if frame is None:
                print(f"Skipping frame {self.frame_index} (file missing)")
                self.next_frame()
                continue

            frame = cv2.resize(frame, (640, 480))
            cv2.imshow(f"Yolo Visualizer", frame)
            key = cv2.waitKey(0) & 0xFF

            if key in [ord('q'), 27]:  # 'q' or ESC to quit
                break
            elif key == ord('d'):
                self.next_frame()
            elif key == ord('a'):
                self.previous_frame()
            elif key == ord('t'):
                self.set_mode(YoloVisualizer.MODE_TRAIN)
            elif key == ord('v'):
                self.set_mode(YoloVisualizer.MODE_VAL)

        cv2.destroyAllWindows()


if __name__ == "__main__":
    dataset_path = r"C:\Users\Raja\Downloads\hackathon2_train_1\train_1"
    classes_path = r"C:\Users\Raja\Downloads\Hackathon2_scripts\Hackathon2_scripts\classes.txt"
    vis = YoloVisualizer(dataset_path, classes_file=classes_path)
    vis.run()
