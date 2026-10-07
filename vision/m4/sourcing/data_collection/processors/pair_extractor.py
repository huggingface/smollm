import logging

from m4.sourcing.data_collection.utils import (
    compute_clip_score,
    fetch_single_image,
    make_selectolax_tree,
    simplify_media_node,
)


logger = logging.getLogger(__name__)


class TextMediaPairsExtractor:
    def __init__(
        self,
        dom_tree_simplificator,
        pre_extraction_simplificator,
        also_extract_images_not_in_simplified_dom_tree=False,
        extract_clip_scores=True,
    ):
        self.dom_tree_simplificator = dom_tree_simplificator
        self.pre_extraction_simplificator = pre_extraction_simplificator
        self.also_extract_images_not_in_simplified_dom_tree = also_extract_images_not_in_simplified_dom_tree
        self.extract_clip_scores = extract_clip_scores

    def __call__(self, html_str, page_url):
        images_in_simplified_dom_tree = self._extraction(html_str, page_url)

        images_not_in_simplified_dom_tree = []
        if self.also_extract_images_not_in_simplified_dom_tree:
            images_not_in_simplified_dom_tree = self._extract_images_not_in_simplified_dom_tree(
                html_str, page_url, images_in_simplified_dom_tree
            )

        return images_in_simplified_dom_tree + images_not_in_simplified_dom_tree

    def _extraction(self, html_str, page_url, diff_level_paths_max_cutoff=5):
        selectolax_tree = self.dom_tree_simplificator(html_str, type_return="selectolax_tree")
        list_nodes = self.pre_extraction_simplificator(selectolax_tree, page_url=page_url)

        images_in_simplified_dom_tree = []
        for ind, node in enumerate(list_nodes):
            if node.tag == "img":
                media_info = node.media_info
                url = media_info["src"]
                image = fetch_single_image(url, timeout=1)
                if image is not None:
                    media_info["original_width"], media_info["original_height"] = image.size
                    if image.format:
                        media_info["format"] = image.format.lower()
                    if ind < len(list_nodes) - 1:
                        if list_nodes[ind + 1].tag == "-text":
                            media_info["extracted_text"] = list_nodes[ind + 1].text.split("\n\n")[0]
                    if self.extract_clip_scores:
                        media_info = self._get_clip_scores(media_info, image)
                    media_info["image_in_simplified_dom_tree"] = True
                    images_in_simplified_dom_tree.append(media_info)
        return images_in_simplified_dom_tree

    def _extract_images_not_in_simplified_dom_tree(self, html_str, page_url, images_in_simplified_dom_tree):
        selectolax_tree = make_selectolax_tree(html_str)
        all_images = [
            selectolax_node for selectolax_node in selectolax_tree.root.traverse() if selectolax_node.tag == "img"
        ]
        all_images = [simplify_media_node(selectolax_node, page_url=page_url) for selectolax_node in all_images]
        all_images = [image for image in all_images if image]

        set_images_in_simplified_dom_tree = set([media_info["src"] for media_info in images_in_simplified_dom_tree])
        images_not_in_simplified_dom_tree = [
            media_info for media_info in all_images if media_info["src"] not in set_images_in_simplified_dom_tree
        ]
        for ind, media_info in enumerate(images_not_in_simplified_dom_tree):
            url = media_info["src"]
            image = fetch_single_image(url, timeout=1)
            if image is not None:
                media_info["original_width"], media_info["original_height"] = image.size
                if image.format:
                    media_info["format"] = image.format.lower()
                if self.extract_clip_scores:
                    media_info = self._get_clip_scores(media_info, image)
                media_info["image_in_simplified_dom_tree"] = False
                images_not_in_simplified_dom_tree[ind] = media_info
            else:
                images_not_in_simplified_dom_tree[ind] = None
        images_not_in_simplified_dom_tree = [image for image in images_not_in_simplified_dom_tree if image]

        return images_not_in_simplified_dom_tree

    def _get_clip_scores(self, media_info, image):
        """If possible, modifies `media_info`to add clip scores on available texts"""
        texts = []
        for text_key in ["formatted_filename", "alt_text", "extracted_text"]:
            if text_key in media_info and media_info[text_key] != "":
                texts.append(media_info[text_key])
        if not texts:
            return media_info

        try:
            clip_scores = compute_clip_score(texts=texts, image=image).tolist()
            idx = 0
            for text_key in ["formatted_filename", "alt_text", "extracted_text"]:
                if text_key in media_info and media_info[text_key] != "":
                    media_info[f"clip_score_image_{text_key}"] = clip_scores[idx]
                    idx += 1
        except ValueError:
            logger.warning(f"ValueError occurred while computing CLIP scores for image ({media_info}). Skipping image.")
        except Exception as exception:
            logger.error(
                f"Error *{exception}* occurred while computing CLIP scores for image ({media_info}). Use at your own"
                " risk."
            )

        return media_info
