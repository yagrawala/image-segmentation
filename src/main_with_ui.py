# ================================================
# Skeleton codes for HW4
# Read the skeleton codes carefully and put all your
# codes into main function
# ================================================
import sys
import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import slic
from skimage.segmentation import mark_boundaries
from skimage.data import astronaut
from skimage.util import img_as_float
import maxflow
from scipy.spatial import Delaunay

old_pt = ()
new_pt = ()
img = []
img_marking = []
fg_counter = 0
bg_counter = 0
fg_max = 2
bg_max = 2
read_pt = False

def help_message():
   print("\n\nUsage: "+sys.argv[0] +" [Input_Image_Path]")
   print("Example:")
   print(sys.argv[0] + " astronaut.png")

# Calculate the SLIC superpixels, their histograms and neighbors
def superpixels_histograms_neighbors(img):
    # SLIC
    segments = slic(img, n_segments=600, compactness=20)
    segments_ids = np.unique(segments)

    # centers
    centers = np.array([np.mean(np.nonzero(segments==i),axis=1) for i in segments_ids])

    # H-S histograms for all superpixels
    hsv = cv2.cvtColor(img.astype('float32'), cv2.COLOR_BGR2HSV)
    bins = [20, 20] # H = S = 20
    ranges = [0, 360, 0, 1] # H: [0, 360], S: [0, 1]
    colors_hists = np.float32([cv2.calcHist([hsv],[0, 1], np.uint8(segments==i), bins, ranges).flatten() for i in segments_ids])

    # neighbors via Delaunay tesselation
    tri = Delaunay(centers)

    return (centers,colors_hists,segments,tri.vertex_neighbor_vertices)

# Get superpixels IDs for FG and BG from marking
def find_superpixels_under_marking(marking, superpixels):
    fg_segments = np.unique(superpixels[marking[:,:,0]!=255])
    bg_segments = np.unique(superpixels[marking[:,:,2]!=255])
    return (fg_segments, bg_segments)

# Sum up the histograms for a given selection of superpixel IDs, normalize
def cumulative_histogram_for_superpixels(ids, histograms):
    h = np.sum(histograms[ids],axis=0)
    return h / h.sum()

# Get a bool mask of the pixels for a given selection of superpixel IDs
def pixels_for_segment_selection(superpixels_labels, selection):
    pixels_mask = np.where(np.isin(superpixels_labels, selection), True, False)
    return pixels_mask

# Get a normalized version of the given histograms (divide by sum)
def normalize_histograms(histograms):
    return np.float32([h / h.sum() for h in histograms])

# Perform graph cut using superpixels histograms
def do_graph_cut(fgbg_hists, fgbg_superpixels, norm_hists, neighbors):
    num_nodes = norm_hists.shape[0]
    # Create a graph of N nodes, and estimate of 5 edges per node
    g = maxflow.Graph[float](num_nodes, num_nodes * 5)
    # Add N nodes
    nodes = g.add_nodes(num_nodes)

    hist_comp_alg = cv2.HISTCMP_KL_DIV

    # Smoothness term: cost between neighbors
    indptr,indices = neighbors
    for i in range(len(indptr)-1):
        N = indices[indptr[i]:indptr[i+1]] # list of neighbor superpixels
        hi = norm_hists[i]                 # histogram for center
        for n in N:
            if (n < 0) or (n > num_nodes):
                continue
            # Create two edges (forwards and backwards) with capacities based on
            # histogram matching
            hn = norm_hists[n]             # histogram for neighbor
            g.add_edge(nodes[i], nodes[n], 20-cv2.compareHist(hi, hn, hist_comp_alg),
                                           20-cv2.compareHist(hn, hi, hist_comp_alg))

    # Match term: cost to FG/BG
    for i,h in enumerate(norm_hists):
        if i in fgbg_superpixels[0]:
            g.add_tedge(nodes[i], 0, 1000) # FG - set high cost to BG
        elif i in fgbg_superpixels[1]:
            g.add_tedge(nodes[i], 1000, 0) # BG - set high cost to FG
        else:
            g.add_tedge(nodes[i], cv2.compareHist(fgbg_hists[0], h, hist_comp_alg),
                                  cv2.compareHist(fgbg_hists[1], h, hist_comp_alg))

    g.maxflow()
    return g.get_grid_segments(nodes)

def RMSD(target, master):
    # Note: use grayscale images only

    # Get width, height, and number of channels of the master image
    master_height, master_width = master.shape[:2]
    master_channel = len(master.shape)

    # Get width, height, and number of channels of the target image
    target_height, target_width = target.shape[:2]
    target_channel = len(target.shape)

    # Validate the height, width and channels of the input image
    if (master_height != target_height or master_width != target_width or master_channel != target_channel):
        print("here")
        return -1
    else:

        total_diff = 0.0;
        dst = cv2.absdiff(master, target)
        dst = cv2.pow(dst, 2)
        mean = cv2.mean(dst)
        total_diff = mean[0]**(1/2.0)

        return total_diff;

def segment_image():
    global old_pt
    global new_pt
    global img
    global img_marking
    old_pt = ()
    new_pt = ()

    # print("here")
    # cv2.imshow('img_marking', img_marking)
    img_copy = img.copy()

    centers, color_hists, superpixels, neighbors = superpixels_histograms_neighbors( img_copy )

    fg_segments, bg_segments = find_superpixels_under_marking(img_marking, superpixels)

    fg_cumulative_hist = cumulative_histogram_for_superpixels(fg_segments, color_hists)
    bg_cumulative_hist = cumulative_histogram_for_superpixels(bg_segments, color_hists)

    norm_hists = normalize_histograms(color_hists)

    graph_cut = do_graph_cut(   (fg_cumulative_hist, bg_cumulative_hist), 
                                (fg_segments, bg_segments),
                                norm_hists,
                                neighbors)
    # print(graph_cut)

    mask = pixels_for_segment_selection(superpixels, np.nonzero(graph_cut) )
    mask = mask.astype('uint8')*255

    cv2.imshow('mask', mask)
    cv2.waitKey(5000)

    img_marking = np.ones(img.shape, np.uint8)*255
    cv2.setMouseCallback('image', captureEvent)

def drawLine(draw_type):
    # print(draw_type)
    global old_pt
    global new_pt
    global img_marking

    if( draw_type=='fg' ):
        color = ( 0, 0, 255)
    if( draw_type=='bg' ):
        color = ( 255, 0, 0)
    # print(old_pt)
    # print(new_pt)
    img_marking = cv2.line(img_marking, old_pt, new_pt, color, thickness=10, lineType=8, shift=0)
    old_pt = new_pt

# mouse callback function
def captureEvent(event,x,y,flags,param):
    global old_pt
    global new_pt
    global fg_counter
    global bg_counter
    global fg_max
    global bg_max
    global read_pt

    # mouse down
    # print(event)
    # print(fg_counter, bg_counter)

    if(event == 1):
        old_pt = (x,y)
        new_pt = (x,y)
        read_pt = True
        if( fg_counter < fg_max):
            fg_counter += 1
        else:
            bg_counter += 1

    # get drag positions
    if( event==0 and read_pt ):
        if( fg_counter <= fg_max and bg_counter==0):
            draw_type = 'fg'
        else:
            draw_type = 'bg'
        new_pt = (x,y)
        drawLine(draw_type)

    # mouse up
    if(event == 4):
        read_pt = False
        if(bg_counter == bg_max):
            fg_counter = 0
            bg_counter = 0
            segment_image()

if __name__ == '__main__':
    # Create a black image, a window and bind the function to window
    global img
    global img_marking

    # validate the input arguments
    if (len(sys.argv) != 2):
        help_message()
        sys.exit()

    img = cv2.imread(sys.argv[1], cv2.IMREAD_COLOR)
    img_marking = np.ones(img.shape, np.uint8)*255
    cv2.namedWindow('image')
    cv2.setMouseCallback('image', captureEvent)

    while(1):
        cv2.imshow('image',img)
        cv2.imshow('img_marking',img_marking)
        if cv2.waitKey(20) & 0xFF == 27:
            break
    cv2.destroyAllWindows()
