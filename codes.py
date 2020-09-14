#
    # frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)) for frame in video]
    # frames=[ImageOps.flip(frame) for frame in frames]
    #

    # frames_tracked = []
    #
    #
    #     # Detect faces
    #
    #
    #     # Draw faces
    #
    #
    #     # Add to frame list
    #     frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    # print('\nDone')
    #
    # dim = frames_tracked[0].size
    # fourcc = cv2.VideoWriter_fourcc(*'FMP4')
    # video_tracked = cv2.VideoWriter('video_tracked.mp4', fourcc, 25.0, dim)
    # for frame in frames_tracked:
    #     video_tracked.write(cv2.cvtColor(np.array(frame), cv2.COLOR_RGB2BGR))
    # video_tracked.release()
    #
    # print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.
    #
