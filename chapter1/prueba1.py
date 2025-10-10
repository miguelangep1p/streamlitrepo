import cv2

img = cv2.imread('./images/persona.jpg', cv2.IMREAD_COLOR)

if img is None:
    print("Error: No se pudo cargar la imagen. Verifica la ruta.")
else:
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    yuv_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
    y, u, v = cv2.split(yuv_img)

    cv2.namedWindow('Grayscale image', cv2.WINDOW_NORMAL)
    cv2.namedWindow('Y channel', cv2.WINDOW_NORMAL)
    cv2.namedWindow('U channel', cv2.WINDOW_NORMAL)
    cv2.namedWindow('V channel', cv2.WINDOW_NORMAL)

    cv2.resizeWindow('Grayscale image', 400, 520)
    cv2.resizeWindow('Y channel', 400, 520)
    cv2.resizeWindow('U channel', 400, 520)
    cv2.resizeWindow('V channel', 400, 520)

    cv2.imshow('Grayscale image', gray_img)
    cv2.imshow('Y channel', y)
    cv2.imshow('U channel', u)
    cv2.imshow('V channel', v)

    cv2.waitKey()
    cv2.destroyAllWindows()
