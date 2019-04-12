import mmcv
import numpy as np
import os
import random
from PIL import Image, ImageDraw, ImageFont, ImageFilter

def create_bg_image(bground_path, width, height):
    bground_list = os.listdir(bground_path)
    bground_choice = random.choice(bground_list)
    bground = Image.open(bground_path+bground_choice)
    bground = bground.resize((width, height))

    return bground

def random_word_color():
    font_color_choice = [[5, 10, 15],[5, 10, 15],[5, 10, 15]]
    font_color = random.choice(font_color_choice)

    noise = np.array([random.randint(0,10),random.randint(0,10),random.randint(0,10)])
    font_color = (np.array(font_color) + noise).tolist()

    #print('font_color：',font_color)

    return tuple(font_color)

def darken_func(image):
    filters = [ImageFilter.GaussianBlur(radius=0.5),
                ImageFilter.SMOOTH_MORE]

    for filter_ in filters:
        image = image.filter(filter_)
    #image = img.resize((290,32))

    return image


def random_x_y(bground_size, font_size, random_x=False):
    width, height = bground_size
    xb = int((width-font_size)/2)
    yb = int((height-font_size)/2)
    #print(bground_size)
    if random_x:
        x = random.randint(0, int(width-font_size))
    else:
        x = random.randint(int(0.9*xb), xb)
    y = random.randint(-yb, -int(0.9*yb))
    # x = y = int(5)

    return x, y

def random_font_size(bground_size, random_size=False):
    if random_size:
        font_size = random.randint(int(0.5*bground_size[1]), int(0.95*bground_size[1]))
    else:
        font_size = random.randint(int(0.75*bground_size[1]), int(0.95*bground_size[1]))

    return font_size

def random_font(font_path, font_list=None):
    if font_list is None:
        font_list = os.listdir(font_path)
    random_font = random.choice(font_list)

    return font_path + random_font

def main():
    cstat_path = '/home/chenriquan/Datasets/CC/cstat.json'
    cstat = mmcv.load(cstat_path)
    vocab = mmcv.load('/home/chenriquan/Datasets/CC/vocab.json')

    hard_limit = 50
    hc_list = []
    for k, v in cstat.items():
        if v < hard_limit:
            # hc_list += [k] * (hard_limit + 20 - v)
            hc_list += [k] * 100
        else:
            hc_list += [k] * 30
        # elif v > 100:
        #     hc_list += [k] * 100
        # elif v >= 30:
        #     hc_list += [k] * 100

    print('hard candidate characters list:', len(hc_list))
    r_ind = np.arange(len(hc_list))
    np.random.shuffle(r_ind)
    hc_list = [hc_list[x] for x in r_ind]

    pbar, upind = mmcv.ProgressBar(len(hc_list)//100), 0

    texture_root = './texture/'
    font_list = os.listdir('./font/')
    output_root = './gen_data/'
    output_anno_path = './gen_anno.pickle'

    ti = 0
    img_id = 1
    output_anno = []
    while ti < len(hc_list):
        random_x_bias, random_size = True, False
        bboxes = []

        c_len = np.random.randint(1, 17)
        if c_len > len(hc_list) - ti:
            c_len = len(hc_list) - ti

        rimg_w = np.random.randint(32, 64)
        rimg_h = int(rimg_w * (np.random.rand()*0.2+0.8))
        font_color = random_word_color()

        # padded texture
        texture_img = create_bg_image(texture_root, rimg_w, int((c_len+0.5)*rimg_h))
        padding_imgx = int((0.2*np.random.rand())*rimg_w)
        new_size = (texture_img.size[0]*2, texture_img.size[1])
        bg_image = Image.new("RGB", new_size)
        bg_image.paste(texture_img, (0, 0))
        bg_image.paste(texture_img, (int(bg_image.size[0]/2), 0))
        minx = int((bg_image.size[0]-padding_imgx*2-rimg_w)/2)
        bg_crop_bbox = (minx, 0, (minx+padding_imgx*2+rimg_w), bg_image.size[1])
        bg_image = bg_image.crop(bg_crop_bbox)

        # draw character one by one
        y_bias = int((0.2*np.random.rand()+0.05)*rimg_h)
        cand_c = []
        if c_len == 4 and rimg_w >= 48: random_size = True
        while y_bias < bg_image.size[1]:
            if ti >= len(hc_list): break
            character = hc_list[ti]

            font_size = random_font_size((rimg_w, rimg_h), random_size)
            draw_x, draw_y = random_x_y((rimg_w, rimg_h), font_size, random_x_bias)
            draw = ImageDraw.Draw(bg_image)

            # loop to choose suitable font
            mask, offset = [], [0, 0]
            while sum(mask) == 0:
                font_name = random_font('./font/', font_list)
                font = ImageFont.truetype(font_name, font_size)
                try:
                    mask, offset = font.getmask2(character, bg_image.mode)
                except AttributeError:
                    mask = font.getmask(character, bg_image.mode)

            if mask.size[1]+int(draw_y)+y_bias > bg_image.size[1]-int(0.2*rimg_h):
                break

            # draw
            draw.text((int(padding_imgx+draw_x), int(draw_y)+y_bias), character, fill=font_color, font=font)
            bx, by = int(draw_x+offset[0]), int(draw_y+offset[1])+y_bias

            font_sizex, font_sizey = mask.size
            padx = 0.05*font_size
            pady = 0.05*mask.size[1]
            x1, y1, x2, y2 = int(bx-padx+padding_imgx), int(by-pady), int(bx+font_sizex+padx+padding_imgx), int(by+font_sizey+pady)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(bg_image.size[0], x2), min(bg_image.size[1], y2)

            # draw.rectangle((x1, y1, x2, y2), outline=(0, 255, 255))
            bboxes.append([x1, y1, x2, y2])

            if random_size:
                y_bias = int(y2+0.2*rimg_h)
            else:
                y_bias = int(y2+(0.08*np.random.rand()+0.05)*rimg_h)
            cand_c.append(character)
            ti += 1

        bg_image = darken_func(bg_image)
        filename = '%06d.jpg'%img_id
        bg_image.save(os.path.join(output_root, filename))
        anno = dict(
            filename=filename,
            width=bg_image.size[0],
            height=bg_image.size[1],
            ann=dict(
                bboxes=np.array(bboxes, dtype=np.float32),
                labels=np.array([1]*len(bboxes), dtype=np.int64),
                texts=cand_c,
                text_labels=np.array([vocab[c] for c in cand_c], dtype=np.int64)
            ),
        )
        output_anno.append(anno)

        img_id += 1

        if ti//100 > upind:
            upind = ti//100
            pbar.update()

    mmcv.dump(output_anno, output_anno_path)



if __name__ == '__main__':
    main()
