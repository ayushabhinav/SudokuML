'''Process image to solve sudoku'''
import sys
from datetime import datetime
import multiprocessing as mp
import cv2
import numpy as np
import pytesseract
from sudoku import  Mode, Sudoku, cell_info


def draw_contour(image, cntrs, cntr_idx, color, thickness=1):
    '''
    draw contour on image
    :param image:
    :param cntrs:
    :param cntr_idx:
    :param color:
    :param thickness:
    :return:
    '''
    for idx in cntr_idx:
        cv2.drawContours(image, cntrs, idx, color=color, thickness=thickness)


def _get_Centroid(cntr):
    '''
    get centroid from contour
    :param cntr:
    :return: cntroid
    '''
    x_centriod, y_centriod = None, None
    try:
        M = cv2.moments(cntr)
        x_centriod = int(M['m10'] / M['m00'])
        y_centriod = int(M['m01'] / M['m00'])
    except ZeroDivisionError:
        pass

    return (x_centriod, y_centriod)



def is_rectangle(cntr):
    '''
    check if contour is rectangle
    :param cntr:
    :return: True/False
    '''
    perimeter = cv2.arcLength(cntr, True)
    approx_cntrs = cv2.approxPolyDP(cntr, 0.03 * perimeter, True)
    count_vertex = len(approx_cntrs)
    if count_vertex == 4:
        return True
    return False



def get_centroids(cell_cntrs):
    '''
    get centroid of each cell
    :param cell_cntrs:
    :return: centroid(x, y, x_bucket, y_bucket, pos_in_cntr)
    '''
    centroids = list()
    x_buckets = dict()
    y_buckets = dict()
    for idx, cntr in enumerate(cell_cntrs):
        x_bucket = None
        y_bucket = None
        x_centroid, y_centroid = _get_Centroid(cntr)
        if x_centroid is None or y_centroid is None:
            continue
        for key in x_buckets.keys():
            if x_centroid in key:
                x_bucket = x_buckets.get(key)
        if x_bucket is None:
            x_bucket = x_centroid
            x_buckets.update({range(x_centroid - 25, x_centroid + 25) : x_centroid})

        for key in y_buckets.keys():
            if y_centroid in key:
                y_bucket = y_buckets.get(key)
        if y_bucket is None:
            y_bucket = y_centroid
            y_buckets.update({range(y_centroid - 25, y_centroid + 25) : y_centroid})
        centroids.append((x_centroid, y_centroid, x_bucket, y_bucket, idx))

    return centroids


def pre_process_image(image):
    '''
    pre_process given image
    :param image:
    :return: processed_image
    '''
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    blur_image = cv2.GaussianBlur(gray_image, (9, 9), 0)
    thresh_image = cv2.adaptiveThreshold(blur_image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                         cv2.THRESH_BINARY_INV, 57, 5)

    kernel = np.array([[0., 1., 0.], [1., 1., 1.], [0., 1., 0.]], dtype=np.uint8)
    thresh_image = cv2.dilate(thresh_image, kernel)


    return thresh_image


def _get_digit(input_image):
    '''
    get digit from given cell image
    :param input_image:
    :return:digit/None
    '''
    i, j, cell_image = input_image
    digit = pytesseract.image_to_string(cell_image,
                                        lang="eng",
                                        config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    if digit.strip() != '':
        return cell_info(i+1, j+1, digit)

    return None


def get_digit_from_image(cell_cntrs, centroids, image, thresh_image):
    '''
    get digit from sudoku
    :param cell_cntrs: cntrs of cell
    :param centroids: centroids
    :param image: image
    :param thresh_image: threshold image
    :return: list of digit - cell_info(i, j, digit)
    '''
    cells = list()
    inputs = list()
    for i in range(0, 9):
        for j in range(0, 9):
            k = 9 * i + j
            cntr = cell_cntrs[centroids[k][-1]]

            x_max, y_max = np.max(np.squeeze(cntr), axis=0)
            x_min, y_min = np.min(np.squeeze(cntr), axis=0)

            cell_image = image[y_min + 10 : y_max - 10, x_min + 10 : x_max - 10]
            thresh_cell_image = thresh_image[y_min + 10 : y_max - 10, x_min + 10 : x_max - 10]

            if any(thresh_cell_image.ravel() == 255):
                inputs.append((i, j, cell_image))

    with mp.Pool(mp.cpu_count()) as pool:
        cells = pool.map(_get_digit, inputs)

    return cells



def solve_sudoku(cells):
    '''
    solve sudoku
    :param cells:
    :return: list of tuple - (digit, True/False)
    '''

    sud = Sudoku(show_steps=False)
    sud.load_data(mode=Mode.CMDLINE, data=cells)
    print('Given Sudoku')
    sud.show()
    sol = None
    if sud.solve():
        print("Solution ---->")
        sud.show()
        sol = sud.get_solution()
    else:
        print('No solution found')

    return sol


def show_solution(image, solution, centroids):
    '''
    put solution on image
    :param image: image
    :param solution: sudoku solution
    :param centroids: cetroids
    :return:
    '''

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = .5
    for idx, sol in enumerate(solution):
        if sol[1]:
            cv2.putText(image,
                        f'{sol[0]}',
                        (centroids[idx][0], centroids[idx][1]),
                        font, font_scale,
                        color=[0, 0, 255],
                        thickness=2)



def process_image(image):
    '''
    process image
    :param image: image
    :return:None
    '''
    start_time = datetime.now()
    print("My Function Called...")

    thresh_image = pre_process_image(image)
    # cv2.imshow('Thresh Image', thresh_image)

    # find Contours List
    cntrs, _ = cv2.findContours(thresh_image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    cntrs = [cntr for cntr in cntrs if is_rectangle(cntr)]
    cntrs = sorted(cntrs, key=cv2.contourArea, reverse=True)

    print(f'len cntr{len(cntrs)}')
    if len(cntrs) < 82:
        return image, False

    cell_cntrs = cntrs[1:82]
    centroids = get_centroids(cell_cntrs)
    centroids.sort(key=lambda x: (x[3], x[2]))

    start_time_get_digit = datetime.now()
    cells = get_digit_from_image(cell_cntrs, centroids, image, thresh_image)
    end_time_get_digit = datetime.now()

    solution = solve_sudoku(cells)

    if solution is not None:
        show_solution(image, solution, centroids)
    else:
        return image, False
    cv2.imshow('Solved Image', image)
    end_time = datetime.now()

    print(f'Processing Time:{end_time - start_time}')
    print(f'Processing Time Get Digit:{end_time_get_digit -  start_time_get_digit}')

    return image, True



if __name__ == '__main__':
    # if len(sys.argv) == 2:
    #     IMAGE_FILE = sys.argv[1]
    # else:
    #     IMAGE_FILE = 'frame_5.png'
    # try:
    #     image = cv2.imread(IMAGE_FILE)
    # except:
    #     print('Image File cannot be loaded')
    # cv2.imshow('Image', image)
    # process_image(image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


    # sys.exit(0)
    # -- Below code is for video processing

    capture = cv2.VideoCapture(1)

    # Check if camera opened successfully
    if capture.isOpened()is False:
        print("Error opening the camera")

    # Read until video is completed

    fps = 10
    # capSize = (1028,720) # this is the size of my source video
    capSize = (640,480) # this is the size of my source video
    # capSize = (640,352) # this is the size of my source video
    fourcc = cv2.VideoWriter_fourcc('m', 'p', '4', 'v') # note the lower case
    vout = cv2.VideoWriter()
    success = vout.open('output.mov',fourcc,fps,capSize,True)

    while capture.isOpened():
        # Capture frame-by-frame from the camera
        ret, frame = capture.read()
        # frame_width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        # frame_height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        # print(frame_width, frame_height)
        # fps = capture.get(cv2.CAP_PROP_FPS)
        # print(fps)
        capture.set(cv2.CAP_PROP_FRAME_WIDTH, 60)
        capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 60)

        if ret is True:
            # Display the captured frame:
            # height, width = frame.shape[:2]
            # M = cv2.getRotationMatrix2D((width / 2.0, height / 2.0), 90, 1)
            # dst_image = cv2.warpAffine(frame, M, (width, height))
            # cv2.imshow('Solved Image', dst_image)
            cv2.imshow('Solved Image', frame)
            try:
                pass
                # # out = cv2.VideoWriter('outpy.avi',cv2.VideoWriter_fourcc('X','V','I','D'), 10, (60,60))
                frame, is_processed = process_image(frame)
                if is_processed:
                    for _ in range(fps * 4):
                        vout.write(frame)
                vout.write(frame)
            except Exception as e:
                print(f'Exception{e}')
            # Press q on keyboard to exit the program
            if cv2.waitKey(20) & 0xFF == ord('q'):
                break

            # cv2.imwrite(f'frames/frame_{datetime.now()}.png', frame)

        # Break the loop
        else:
            break

    # Release everything:
    capture.release()
    vout.release()
    cv2.destroyAllWindows()
