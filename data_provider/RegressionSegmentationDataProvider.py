import os
import logging
import numpy as np
import cv2
import gc
from tqdm import tqdm
from . import SemanticSegmentationDataProvider
from ..kutils import utilites


class RegressionSegmentationDataProvider(SemanticSegmentationDataProvider):
    def get_scaled_image(self, i, sc_factor):
        image, _ = self._get_src_item(i)

        # Save original image params
        src_image_shape = image.shape

        # Scale image(and mask) if it necessary
        if sc_factor != 1.0:
            # For downscale the best interpolation - INTER_AREA, for upscale - INTER_CUBIC
            img_interp = cv2.INTER_CUBIC if sc_factor > 1.0 else cv2.INTER_AREA

            # For downscale the best interpolation INTER_AREA
            image = cv2.resize(image.astype(np.float32), (0, 0),
                               fx=sc_factor, fy=sc_factor, interpolation=img_interp).astype(image.dtype)
            gc.collect()

        # This filter has been added after training process to obtain better accuracy on high-contrast areas
        # and predict over-bright rocks. Tests proved that this filter grows the accuracy as for little rocks as
        # for big rocks. Should be noticed that at the end, filter for big rocks could damage results by merging piles
        # of little rocks. To fix this issue bigger threshold value should be used for big rocks.
        apply_clahe = True  # (sc_factor >= 1.0)
        if apply_clahe:
            if image.shape[2] > 1:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
                # Median-3 definitely good for scale 0.25 aspecially with d4-group.
                # But for scale 1.0 it is uncertain - some projects like
                # "MuckPileDatasets.unseen\airzaar\13007.qa7966" predicts little rocks good(best) without any filter
                # before, but have "left-edge-bright" artifact(which partially solved by smoothing between patches),
                # but some other "2021.04.06\3GSM.comparison\dev-oktai\7223" predicts better when median kernel 3 or
                # even 5. So remain it here for both scales with kernel 3 as compromise between NO-filter and Median-5.
                gray = cv2.medianBlur(gray, 3)

            if False:  # play
                # https://stackoverflow.com/questions/44047819/increase-image-brightness-without-overflow/44054699#44054699

                # Dilate the image, in order to get rid of the text. This step somewhat helps to preserve the bar code.
                dilated_img = cv2.dilate(gray, np.ones((7, 7), np.uint8))
                cv2.imwrite('dilated_img.png', dilated_img)

                # Median blur the result with a decent sized kernel to further suppress any text.
                # This should get us a fairly good background image that contains all the shadows and/or discoloration.
                bg_img = cv2.medianBlur(dilated_img, 21)
                cv2.imwrite('bg_img.png', bg_img)

                # Calculate the difference between the original and the background we just obtained.
                # The bits that are identical will be black (close to 0 difference), the text will be white
                # (large difference). Since we want black on white, we invert the result
                diff_img = 255 - cv2.absdiff(gray, bg_img)
                cv2.imwrite('diff_img.png', diff_img)

                # Normalize the image, so that we use the full dynamic range.
                norm_img = diff_img.copy()  # Needed for 3.x compatibility
                cv2.normalize(diff_img, norm_img, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1)
                cv2.imwrite('norm_img.png', norm_img)

                gray = norm_img

            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))  # 1
            # clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))  # 2
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))  # 3
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(16, 16))  # 4
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))  # 5
            # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(64, 64))  # 6
            # clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(64, 64))  # 7 +
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(128, 128))  # 8
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(256, 256))  # 9
            # clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(512, 512))  # 10  less contrast
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(512, 512))  # 11  the same like 9
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(256, 256))  # 12  almost like 9 but a little bit +
            # clahe = cv2.createCLAHE(clipLimit=4.0, tileGridSize=(128, 128))  # 13  # as + as - w.r.t 9
            # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(1024, 1024))  # 14 very bad
            # clahe = cv2.createCLAHE(clipLimit=5.0, tileGridSize=(64, 64))  # 15  # as + as - w.r.t 9
            # clahe = cv2.createCLAHE(clipLimit=128.0, tileGridSize=(256, 256))  # 16  idea - better than 12. BEST
            # clahe = cv2.createCLAHE(clipLimit=192.0, tileGridSize=(256, 256))  # 17 - copy of 16
            # clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(256, 256))  # 18. a little bit better than 16.
            # clahe = cv2.createCLAHE(clipLimit=8.0, tileGridSize=(256, 256))  # 19 +/i w.r.t 18
            clahe = cv2.createCLAHE(clipLimit=16.0, tileGridSize=(int(256*sc_factor), int(256*sc_factor)))  # 18s. GOOD but Big predicted rocks merge little rocks to one rock.
            gray = clahe.apply(gray)
            del clahe
            gc.collect()

            image = np.dstack([gray, gray, gray])

        image, _ = self._apply_prepproc(image, None)

        return src_image_shape, image

    def _preprocess_mask(self, mask):
        mask = mask.astype(np.float32) / 255.0
        if len(mask.shape) == 2:
            mask = mask[..., np.newaxis]
        return mask

    def show(self, i):
        image, mask = self.__getitem__(i)

        # image_rgb = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)
        image_rgb = image[..., :3].copy()
        mask = (mask*255).astype(np.uint8)

        utilites.visualize(
            title=os.path.basename(self.get_fname(i)),
            img_fname=None,
            Image=image_rgb,
            Masked_Image=((image_rgb.astype(np.float32) +
                           np.dstack((mask, mask*0, mask*0)).astype(np.float32))//2).astype(np.uint8),
        )

    def show_predicted(self, solver, show_random_items_nb, save_imgs=False):
        ids = np.random.choice(np.arange(len(self)), size=show_random_items_nb, replace=False)
        result_list = list()
        for i in tqdm(ids):
            image, gt_mask = self.__getitem__(i)
            image = np.expand_dims(image, axis=0)
            pr_mask = solver.model.predict(image, verbose=0)[0]
            pr_mask = solver.post_predict(pr_mask)
            scores = solver.model.evaluate(image, np.expand_dims(gt_mask, axis=0), batch_size=1, verbose=0)

            # gt_cntrs = utilites.get_contours((gt_mask * 255).astype(np.uint8))
            # pr_cntrs = utilites.get_contours((pr_mask * 255).astype(np.uint8))
            img_metrics = dict()
            for metric, value in zip(solver.metrics, scores[1:]):
                metric_name = metric if isinstance(metric, str) else metric.__name__
                img_metrics[metric_name] = value

            item = dict({'index': i, 'metrics': img_metrics})
            item['gt_mask'] = gt_mask.squeeze()
            item['pr_mask'] = pr_mask.squeeze()
            item['image'] = image.squeeze()
            result_list.append(item)
        # sort list to start from the worst result
        result_list = sorted(result_list, key=lambda it: it['metrics']['mae'])[::-1]  # todo: why hardcoded mae ?

        img_storing_dir = os.path.join(self.conf.solution_dir, 'evaluate_imgs')
        if not os.path.isdir(img_storing_dir) and save_imgs:
            os.makedirs(img_storing_dir)
            logging.info('Folder {} has been created'.format(img_storing_dir))

        for item in result_list:
            image = item['image']
            img_fname = os.path.basename(self.get_fname(item['index']))

            # gt_mask = item['gt_mask']
            pr_mask = item['pr_mask']

            img_temp = (utilites.denormalize(image[..., :3]) * 255).astype(np.uint8)

            pr_mask = (pr_mask * 255).astype(np.uint8)

            utilites.visualize(
                title='{}, MAE:{:.4f}'.format(img_fname, item['metrics']['mae']),
                img_fname=os.path.join(img_storing_dir, img_fname) if save_imgs else None,
                Image=img_temp,
                Masked_Image=((img_temp.astype(np.float32) +
                               np.dstack((pr_mask, pr_mask*0, pr_mask*0)).astype(np.float32))//2).astype(np.uint8),
                Mask=pr_mask
            )


class RegressionSegmentationSingleDataProvider(RegressionSegmentationDataProvider):
    # todo: seems copy of other class SemanticSegmentationSingleDataProvider
    def __init__(self, data_reader, img_fname, himg_fname, bbox, configure, prep_getter):
        super().__init__(data_reader=data_reader,
                         data_dir='',
                         ids=list(),
                         bbox=bbox,
                         conf=configure,
                         min_mask_ratio=0.0,
                         augmentation=None,
                         prep_getter=prep_getter)

        self.src_folders = ['1', '2']  # In this case folder names are no mater. The order is matter as always.
        self.src_data = dict({k: list() for k in self.src_folders})

        self.src_data[self.src_folders[0]].append(img_fname)
        self.src_data[self.src_folders[1]].append(himg_fname)

        # Find the actual length of dataset
        keys = list(self.src_data)
        self._length = len(self.src_data[keys[0]]) if len(keys) > 0 else 0
