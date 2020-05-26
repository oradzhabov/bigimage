import numpy as np


def get_rgb_from_i(rgb_int):
    blue = rgb_int & 255
    green = (rgb_int >> 8) & 255
    red = (rgb_int >> 16) & 255
    return red, green, blue


def get_i_from_rgb(rgb):
    red = rgb[0]
    green = rgb[1]
    blue = rgb[2]
    return (red << 16) + (green << 8) + blue


class LookupTable(object):
    AL_UNIVERSAL = 0
    AL_ONE_CHANNEL = 1

    def __init__(self, palette, algo=AL_UNIVERSAL):
        if isinstance(palette, str):
            palette = self.load_palette(palette)
        self.__cache = None
        self.make_cache(palette, algo)

    @staticmethod
    def load_palette(filename):
        def to_perc(v):
            if "%" in v:
                return int(v.replace("%", ""))
            elif v == "nv":
                return 0

        palette = []
        with open(filename, 'r') as fr:
            for line in fr:
                if line:
                    p, r, g, b = line.strip().split()[:4]
                    palette.append((to_perc(p), int(r), int(g), int(b)))
        return palette

    def make_cache(self, palette, algo):
        def get_current(vc, v1, v2):
            if v1 is None or v2 is None:
                return None
            return (v2 - v1) * vc + v1

        def to_uint8(_val):
            if _val is None:
                return None
            return np.uint8(round(_val))
            
        self.__cache = np.full((256, 256, 256), np.nan, dtype=np.uint8)

        # Algo 1. Universal but slower
        if algo == LookupTable.AL_UNIVERSAL:
            for i in range(len(palette)-1):
                perc1, r1, g1, b1 = palette[i]
                perc2, r2, g2, b2 = palette[i+1]
                range_r = list(range(r1, r2, 1 if r1 < r2 else -1))
                range_g = list(range(g1, g2, 1 if g1 < g2 else -1))
                range_b = list(range(b1, b2, 1 if b1 < b2 else -1))

                range_size = max(len(range_r), len(range_g), len(range_b))
                brange = zip(*(
                        range(range_size),
                        range_r if range_r else [r1] * range_size,
                        range_g if range_g else [g1] * range_size,
                        range_b if range_b else [b1] * range_size,
                        ))
                    
                for v, r, g, b in brange:
                    perc = float(v) / float(range_size)
                    val = get_current(perc, perc1, perc2)
                    if val is not None:
                        self.__cache[r, g, b] = to_uint8(val)

        # Algo 2. Works only when changed only one channel between rows in palette
        elif algo == LookupTable.AL_ONE_CHANNEL:
            for i in range(len(palette)-1):
                perc1, r1, g1, b1 = palette[i]
                perc2, r2, g2, b2 = palette[i+1]
                range_r = list(range(r1, r2, 1 if r1 < r2 else -1))
                range_g = list(range(g1, g2, 1 if g1 < g2 else -1))
                range_b = list(range(b1, b2, 1 if b1 < b2 else -1))
    
                if range_r:
                    g = g1
                    b = b1
                    for r in range_r:
                        perc = float(r - r1) / float(r2 - r1)
                        val = get_current(perc, perc1, perc2)
                        if val is not None:
                            self.__cache[r, g, b] = to_uint8(val)
                if range_g:
                    r = r1
                    b = b1
                    for g in range_g:
                        perc = float(g - g1) / float(g2 - g1)
                        val = get_current(perc, perc1, perc2)
                        if val is not None:
                            self.__cache[r, g, b] = to_uint8(val)
                if range_b:
                    r = r1
                    g = g1
                    for b in range_b:
                        perc = float(b - b1) / float(b2 - b1)
                        val = get_current(perc, perc1, perc2)
                        if val is not None:
                            self.__cache[r, g, b] = to_uint8(val)
        else:
            raise Exception("Unsupported algorithm")

    def __getitem__(self, color_value):
        if isinstance(color_value, np.ndarray):
            if len(color_value[0]) == 3:
                r, g, b = color_value[:, 2], color_value[:, 1], color_value[:, 0]
                return self.__cache[r, g, b]
        if isinstance(color_value, tuple) or isinstance(color_value, list):
            if len(color_value) == 3: 
                r, g, b = color_value
                return self.__cache[r, g, b]
        if isinstance(color_value, int):
            r, g, b = get_rgb_from_i(color_value)
            return self.__cache[r, g, b]
        raise ValueError("Unsupported value")
    
    def __call__(self, v):
        return self.__getitem__(v.tolist())


def color2height(filename_palette, img_bgr):
    lt = LookupTable(filename_palette, LookupTable.AL_ONE_CHANNEL)

    h = img_bgr.shape[0]
    w = img_bgr.shape[1]
    gray = np.zeros(shape=(h, w), dtype=np.uint8)

    """
    for y in range(h):
        row = img_bgr[y]
        rowg = gray[y]
        for x in range(w):
            bgr = row[x]
            rowg[x] = lt[bgr[2], bgr[1], bgr[0]]
    """
    x_arr = np.arange(w)
    for y in range(h):
        row = img_bgr[y]
        rowg = gray[y]
        rowg[x_arr] = lt[row[x_arr]]

    gray = gray/100*255
    # gray = cv2.GaussianBlur(gray,(3,3),0)

    # cv2.imwrite('gray.png', gray)
    return gray
