import cv2
import numpy as np
import sys

DEBUG_INTERACTIVE = True

def nothing(x):
    pass  

def run_debug(img_path):
    
    img = cv2.imread(img_path)
    if img is None:
        print("Image not found.")
        return

    gray_original = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    cv2.namedWindow("Controls")
    cv2.namedWindow("Result",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Blurred",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Edges",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Mask After Morph",cv2.WINDOW_NORMAL)
    cv2.namedWindow("Bilateral Filter",cv2.WINDOW_NORMAL)
    

    # Trackbars
    cv2.createTrackbar("Gaussian", "Controls", 5, 25, nothing)
    cv2.createTrackbar("Canny Low", "Controls", 50, 255, nothing)
    cv2.createTrackbar("Canny High", "Controls", 150, 255, nothing)
    cv2.createTrackbar("Morph", "Controls", 3, 25, nothing)
    cv2.createTrackbar("Bilateral_d", "Controls", 5, 20, nothing)
    cv2.createTrackbar("SigmaColor", "Controls", 10, 200, nothing)
    cv2.createTrackbar("SigmaSpace", "Controls", 20, 200, nothing)

    while True:

        g = cv2.getTrackbarPos("Gaussian", "Controls")
        low = cv2.getTrackbarPos("Canny Low", "Controls")
        high = cv2.getTrackbarPos("Canny High", "Controls")
        m = cv2.getTrackbarPos("Morph", "Controls")
        d = cv2.getTrackbarPos("Bilateral_d", "Controls")
        sigmaColor = cv2.getTrackbarPos("SigmaColor", "Controls")
        sigmaSpace = cv2.getTrackbarPos("SigmaSpace", "Controls")

        # d must be > 0
        if d < 1:
            d = 1
            
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
        
        filtered = cv2.bilateralFilter(
        gray_original,
        d,
        sigmaColor,
        sigmaSpace
        )
        
        edges = cv2.Canny(filtered, low, high)

        kernel = np.ones((m, m), np.uint8)
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel)
        
        #cv2.resizeWindow("Blurred", 800, 600)
        cv2.imshow("Blurred", blurred)
        #cv2.resizeWindow("Edges", 800, 600)
        cv2.imshow("Edges", edges)
        #cv2.resizeWindow("Mask After Morph", 800, 600)
        cv2.imshow("Mask After Morph", mask)
        #cv2.resizeWindow("Bilateral Filter", 800, 600)
        cv2.imshow("Bilateral Filter", filtered)
        
        # Draw contours for visualization
        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_TREE,
            cv2.CHAIN_APPROX_SIMPLE
        )

        debug_img = gray_original.copy()
        cv2.drawContours(debug_img, contours, -1, (0, 255, 0), 1)

        cv2.resizeWindow("Result", 800, 600)
        cv2.imshow("Result", debug_img)

        key = cv2.waitKey(1) & 0xFF

        if key == 27:  # ESC to quit
            break

        if key == ord('s'):  # Press S to print values
            print("\nCurrent values:")
            print(f"Gaussian: {g}")#Low value → noisy edges; High value → softer, fewer edges; Too high → details are lost
            print(f"Canny Low: {low}")#Low threshold ↓ → more edges
            print(f"Canny High: {high}")#High threshold ↑ → fewer edges
            print(f"Morph: {m}")#Small kernel → fragmented contours; Large kernel → thick, merged blobs; Too large → one big blob
            print(f"Bilateral Filter d: {d}")#Size of the pixel neighborhood. Larger = stronger smoothing. Too large = slow + over-smoothed.
            print(f"Bilateral Filter SigmaColor: {sigmaColor}")#How much color difference is tolerated. Low → preserves strong edges. High → more smoothing across edges
            print(f"Bilateral Filter SigmaSpace: {sigmaSpace}")#How far pixels influence each other spatially. Low → very local smoothing. High → broader smoothing
            cv2.destroyAllWindows()


if __name__ == "__main__":
    #User writes path in the console
    #if len(sys.argv) < 2:
    #    print("Usage: python debug_run.py <image_path>")
    #    sys.exit(1)
    #image_path = sys.argv[1]
    
    #Hardcoded path for testing
    image_path = r"C:\Users\esper\Desktop\Trabalho\3D-gen\3D-gen\input\monument_01\20260212_094859.jpg"
    
    run_debug(image_path)
        