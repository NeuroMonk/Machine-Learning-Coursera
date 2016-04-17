import numpy as np

def displayData(X, example_width=None):

    if not example_width:
        example_width = np.round(np.sqrt(X.shape[1])).astype(np.int64)
    

    m, n = X.shape
    example_height = (n / example_width).astype(np.int64)

    display_rows = np.floor(np.sqrt(m)).astype(np.int64)
    display_cols = np.ceil(m/display_rows).astype(np.int64)

    print display_rows, display_cols
    
    pad = 1

    display_array = - np.ones((pad + display_rows * (example_height + pad),\
                               pad + display_cols * (example_width + pad)))

    curr_ex = 1
    for j in range(display_rows):
        for i in range(display_cols):
            if curr_ex > m:
                break

            max_val = np.max(np.abs(X[curr_ex, :]))
            display_array[pad + (j - 1) * (example_height + pad) + range(1, example_height),\
                          pad + (i - 1) * (example_width + pad) + range(1, example_width)] =\
                          np.reshape(X[curr_ex, :], example_height, example_width) / max_val
            
            curr_ex = curr_ex + 1

        if curr_ex>m:
            break