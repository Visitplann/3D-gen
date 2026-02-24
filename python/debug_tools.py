import cv2
import numpy as np
import sys

DEBUG_INTERACTIVE = True

def nothing(x):
    pass  

def run_debug(img_path):
    
    #valid_extensions=('.jpg', '.jpeg', '.png', '.bmp', '.tiff')
    
    #try:
    #    image_files = sorted([f for f in os.listdir(monument_path) if f.lower().endswith(valid_extensions)])
    #
    #    if not image_files:
    #        raise FileNotFoundError(f"Nenhuma imagem encontrada em: {monument_path}")
    #  
    #    for file_name in image_files:
    #
    #        img_path = os.path.join(monument_path, file_name)
    #        print(f"Processando: {file_name}...")

    #       try:
    #            img = cv2.imread(img_path)
        
            #FAILSAFE
    #       if img is None:
    #            print(f"Aviso: {file_name} não é uma imagem válida. A passar á frente...")
    #            continue
    
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found.")
        return

    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Controls")
    cv2.namedWindow("Result")

    # Trackbars
    cv2.createTrackbar("Gaussian", "Controls", 5, 25, nothing)
    cv2.createTrackbar("Canny Low", "Controls", 50, 255, nothing)
    cv2.createTrackbar("Canny High", "Controls", 150, 255, nothing)
    cv2.createTrackbar("Morph", "Controls", 3, 25, nothing)

    while True:

        g = cv2.getTrackbarPos("Gaussian", "Controls")
        low = cv2.getTrackbarPos("Canny Low", "Controls")
        high = cv2.getTrackbarPos("Canny High", "Controls")
        m = cv2.getTrackbarPos("Morph", "Controls")

        # Ensure odd kernel sizes
        if g < 1:
            g = 1
        if g % 2 == 0:
            g += 1

        if m < 1:
            m = 1
        if m % 2 == 0:
            m += 1

        # Processing
        blurred = cv2.GaussianBlur(gray_original, (g, g), 0)
        edges = cv2.Canny(blurred, low, high)

        kernel = np.ones((m, m), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)

        # Draw contours for visualization
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        debug_img = img.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)

        cv2.imshow("Result", debug_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break

        if key == ord('s'):  # Press S to print values
            print("\nCurrent values:")
            print(f"Gaussian: {g}")
            print(f"Canny Low: {low}")
            print(f"Canny High: {high}")
            print(f"Morph: {m}")

    cv2.destroyAllWindows()


if __name__ == "__main__":

    if len(sys.argv) < 2:
        print("Usage: python debug_tuner.py image_path")
    else:
        run_debug(sys.argv[1])
        