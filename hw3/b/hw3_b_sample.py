import cv2
import sys


def hw3_b(img):
    img = cv2.imread(img, cv2.IMREAD_UNCHANGED)
    if img is None:
        print("Error: Image not found")
        return

    # convert to grayscale to threshold
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # threshold to binary image
    _, thresh = cv2.threshold(img_gray, 50, 255, cv2.THRESH_BINARY)

    # find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

    cv2.drawContours(img, contours, -1, (0, 255, 0), 1)

    # display the image with found contours
    while True:
        try:
            cv2.imshow("Contours", img)
            key = cv2.waitKey(1)

            # exit on "q" or "ESC" key press
            if key == ord("q") or key == 27:
                break

        except KeyboardInterrupt:
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python3 hw3_b.py <input image>")
        sys.exit(1)

    hw3_b(sys.argv[1])
