# image-segmentation
Semi-assisted segmentation of image using SLIC(super-pixel) and graph-cut minimisation techniques. 

We calculate the super-pixels and then use user markings to differentiate the super-pixels.
Output is a binary mask, based on the user-markings.

The same is done interactively(user draws markings when code is running) and non-interactively(take markings from an image).
