from mpi4py import MPI
import skvideo.io
import skvideo.datasets
import numpy as np
import cv2

comm = MPI.COMM_WORLD
size = comm.Get_size()
rank = comm.Get_rank()
# initializing subtractor
fgbg = cv2.createBackgroundSubtractorMOG2()

if rank == 0:
    # read video frame-by-frame
    videogen = skvideo.io.vread(skvideo.datasets.bigbuckbunny())
    # split array evenly among processes
    chunks = np.array_split(np.array(videogen), size - 1, axis=0)
    skvideo.io.vwrite("bigbuckbunny.mp4", videogen)

    # send chunks to processors
    for i in range(size - 1):
        chunk = chunks[i]
        comm.send(chunk, dest=i + 1)

    output = np.empty((0, 720, 1280, 1))
    # receive images from processors
    for i in range(1, size):
        frames_chunk = comm.recv(source=i)
        output = np.append(output, frames_chunk, axis=0)

    skvideo.io.vwrite("outputvideo.mp4", output)
else:
    frames_chunk = comm.recv(source=0)
    processed_frames = np.empty((0, 720, 1280, 1))
    for frame in frames_chunk:
        fgmask = fgbg.apply(frame)
        temp = np.expand_dims(fgmask, axis=(0, 3))
        processed_frames = np.append(processed_frames, temp, axis=0)
    comm.send(processed_frames, dest=0)