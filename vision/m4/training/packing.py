"""
This file defines the data packing logic.
"""
import logging
import math
import random
from typing import List

import numpy as np
import torch

from m4.training.utils import END_OF_UTTERANCE_TOKEN, FAKE_TOKEN_AROUND_IMAGE_V2, IMAGE_TOKEN, image_splitting


logger = logging.getLogger(__name__)


# Hyper-parameters
_IMAGE_BONUS_VALUE = 2  # The bonus value for tokens preceding the image token
_MIN_LENGTH_DOCUMENTS_TO_PACK = (
    5  # Minimum lengths of documents to pack together (lengths are measured in number of tokens)
)
RANDOM_LINE_BREAK_PROB = 0.05


def get_splitted_images_and_corresponding_text(
    image,
    vision_encoder_max_image_size,
    max_image_size,
    pre_split_scale_up_max,
    pre_split_scale_up_frequency,
    image_seq_len,
    scale_up_factor=None,
):
    splitted_images_array, image_rows, image_cols = image_splitting(
        image=image,
        vision_encoder_max_image_size=vision_encoder_max_image_size,
        max_image_size=max_image_size,
        pre_split_scale_up_max=pre_split_scale_up_max,
        pre_split_scale_up_frequency=pre_split_scale_up_frequency,
        scale_up_factor=scale_up_factor,
    )
    if len(splitted_images_array) > 1:
        text_splitted_images = ""
        for n_h in range(image_rows):
            for n_w in range(image_cols):
                text_splitted_images += (
                    f"{FAKE_TOKEN_AROUND_IMAGE_V2}"
                    + f"<row_{n_h + 1}_col_{n_w + 1}>"
                    + f"{IMAGE_TOKEN}" * image_seq_len
                )
            text_splitted_images += "\n"
        text_splitted_images += (
            f"\n{FAKE_TOKEN_AROUND_IMAGE_V2}"
            + "<global-img>"
            + f"{IMAGE_TOKEN}" * image_seq_len
            + f"{FAKE_TOKEN_AROUND_IMAGE_V2}"
        )
    else:
        text_splitted_images = (
            f"{FAKE_TOKEN_AROUND_IMAGE_V2}"
            + "<global-img>"
            + f"{IMAGE_TOKEN}" * image_seq_len
            + f"{FAKE_TOKEN_AROUND_IMAGE_V2}"
        )
    return splitted_images_array, text_splitted_images


def remove_extra_images(
    input_ids_: List[int],
    images_: List[torch.FloatTensor],
    max_num_images: int,
    image_seq_len: int,
    fake_token_around_image_id: int,
    image_token_id: int,
    double_breaking_lines_token_ids: List[int],
):
    """
    Removes images if there are more than `max_num_images`.
    Removes the associated image tokens from the text.

    Strategy:
    -Remove all image patterns from the input that correspond to the images after `max_num_images`.
    An image pattern is `<fake_token_around_image><image>...<image>` **without** the ending
    `<fake_token_around_image>`. This way, if two or more images are consecutive, we remove
    `<fake_token_around_image><image>...<image><fake_token_around_image><image>...<image>` but we
    still leave one `<fake_token_around_image>`. For the case of an image alone, we are still left
    with only one `<fake_token_around_image>`.
    -Then, we replace these remaining `<fake_token_around_image>` by `\n\n` because they limit two
    different paragraphs (previously separated by an image) that shouldn't be merged.
    -Since the number of tokens for "\n\n" will be always 2 and we remove at least 2 tokens
    (`<fake_token_around_image><image>`) for each removed image even in the case of the extreme
    limit `image_seq_len=1`, we never have in the end more tokens than what the inputs had.

    Args details:
    `images_` -> Each tensor is of size (3, im_height, im_width)
    """
    num_images_to_discard = len(images_) - max_num_images
    if num_images_to_discard <= 0:
        raise ValueError("We shouldn't use `remove_extra_images` if there isn't any images to remove")

    init_len_input_ids = len(input_ids_)

    images_ = images_[:max_num_images]

    starting_pos_image_patterns = [
        idx
        for idx in range(len(input_ids_[:-1]))
        if (input_ids_[idx] == fake_token_around_image_id) and (input_ids_[idx + 1] == image_token_id)
    ]
    # Sanity check for the input
    if len(starting_pos_image_patterns) != len(images_) + num_images_to_discard:
        raise ValueError("Mismatch in image counting")

    for idx in range(1, num_images_to_discard + 1):
        start_pos = starting_pos_image_patterns[-idx]
        if (
            input_ids_[start_pos : start_pos + image_seq_len + 1]
            != [fake_token_around_image_id] + [image_token_id] * image_seq_len
        ):
            raise ValueError("Wrong tokens in the image sequence")
        del input_ids_[start_pos : start_pos + image_seq_len + 1]

    pos_remaining_fake_tokens = [
        idx
        for idx in range(starting_pos_image_patterns[-num_images_to_discard], len(input_ids_))
        if input_ids_[idx] == fake_token_around_image_id
    ]
    # We replace all tokens in pos_remaining_fake_tokens by [token], so that we replace just after
    # the remaining fake tokens by the list of tokens corresponding to "\n\n". All we need to do
    # to finish is enroll the sublists.
    input_ids_ = [[tok] for tok in input_ids_]
    for pos in pos_remaining_fake_tokens:
        input_ids_[pos] = double_breaking_lines_token_ids
    input_ids_ = [sub_el for el in input_ids_ for sub_el in el]

    # Sanity checks for the output
    starting_pos_image_patterns = [
        idx
        for idx in range(len(input_ids_[:-1]))
        if (input_ids_[idx] == fake_token_around_image_id) and (input_ids_[idx + 1] == image_token_id)
    ]
    if len(starting_pos_image_patterns) != max_num_images:
        raise ValueError("Mismatch in the number of images")
    if len(images_) != max_num_images:
        raise ValueError("Mismatch in the number of images")
    if len(input_ids_) > init_len_input_ids:
        raise ValueError("The returned input_ids shouldn't be longer than the input one")

    return input_ids_, images_


def greedy_packing(
    input_ids_to_pack: List[List[int]],
    images_to_pack: List[List[torch.FloatTensor]],
    max_seq_len: int,
    max_num_images: int,
    image_seq_len: int,
    pad_token_id: int,
    fake_token_around_image_id: int,
    image_token_id: int,
    double_breaking_lines_token_ids: List[int],
    output_input_ids: List[torch.IntTensor] = [],
    output_images: List[torch.FloatTensor] = [],
    output_attention_masks: List[torch.IntTensor] = [],
    output_pixel_attention_masks: List[torch.BoolTensor] = [],
    output_num_images: List[int] = [],
    output_num_text_tokens: List[int] = [],
    truncate_images_within_same_example: bool = False,
    mask_labels: bool = False,
    end_of_utterance_token_id: int = None,
    bos_token_id: int = None,
    eos_token_id: int = None,
    assistant_token_ids: List[int] = None,
):
    """
    Args details:
    `images_to_pack` -> # Each tensor is of size (3, im_height, im_width)
    `output_input_ids` -> # Each tensor is of size (max_seq_len,)
    `output_images` -> # Each tensor is of size (max_num_images, 3, max_sample_height, max_sample_width)
    `output_attention_masks` -> # Each tensor is of size (max_seq_len,)
    `output_pixel_attention_masks` -> # Each tensor is of size (max_num_images, max_sample_height, max_sample_width)
    """
    # We pack the samples with a greedy approach, without cutting any sample in the middle.
    # We start with the first sample. We append to it the second, the third, ..., until we
    # can't add the next one because it would make the text longer than `max_seq_len`.
    # So we create another input for the batch and add the sample here instead. For the next
    # sample, we still check if we could fit it in the previous batch examples. If not, we create
    # another batch example, and so on.

    # Sanity checks
    if len(input_ids_to_pack) != len(images_to_pack):
        raise ValueError("Mismatch lengths of lists")
    if not all([len(input_ids_) <= max_seq_len for input_ids_ in input_ids_to_pack]):
        raise ValueError("All input_ids should be shorter than max_seq_len")

    batch = []
    for input_ids_, images_ in zip(input_ids_to_pack, images_to_pack):
        len_sample = len(input_ids_)
        num_images_in_sample = len(images_)
        win_tetris = False
        for i in range(len(batch)):
            condition_extend_batch_example = len(batch[i][0]) + len_sample <= max_seq_len
            if not truncate_images_within_same_example:
                # For some datasets, we don't want to add sequences containing images that we are removing after
                # because of max_num_images. It would mean we train on the text or the captions without
                # the images. This can fine to do this for OBELICS, but not for PMD.
                # However, in the context of the image splitting strategy, it is generally not safe to do that.
                condition_extend_batch_example = condition_extend_batch_example and (
                    len(batch[i][1]) + num_images_in_sample <= max_num_images
                )
            if condition_extend_batch_example:
                batch[i][0].extend(input_ids_)
                batch[i][1].extend(images_)
                win_tetris = True
                break
        if not win_tetris:
            # images_ is a torch.stack of some images. When we are doing extend on a list
            # and try to add a torch.stack, it will add the the lists the elements inside
            # the torch.stack. Same for input_ids_. Therefore, the following lines are
            # different than doing batch.append(([input_ids_], [images_]))
            new_ex = ([], [])
            # If an example would have more images than max_num_images, we drop it
            # if not truncate_images_within_same_example.
            if truncate_images_within_same_example or (len(images_) <= max_num_images):
                new_ex[0].extend(input_ids_)
                new_ex[1].extend(images_)
                batch.append(new_ex)

    for i in range(len(batch)):
        input_ids_ = batch[i][0]
        images_ = batch[i][1]

        if truncate_images_within_same_example:
            # First, we remove some images from the batch examples if there are
            # more than max_num_images
            num_images_to_discard = len(images_) - max_num_images
            if num_images_to_discard > 0:
                input_ids_, images_ = remove_extra_images(
                    input_ids_=input_ids_,
                    images_=images_,
                    max_num_images=max_num_images,
                    image_seq_len=image_seq_len,
                    fake_token_around_image_id=fake_token_around_image_id,
                    image_token_id=image_token_id,
                    double_breaking_lines_token_ids=double_breaking_lines_token_ids,
                )
        else:
            if len(images_) > max_num_images:
                raise ValueError("We should have fewer images than `max_num_images`")

        # We have hanging issues for examples without any images
        if len(images_) == 0:
            continue

        # Then, we pad and prepare the final tensors
        # First we pad the input_ids
        len_sample = len(input_ids_)
        num_tokens_for_images = len([tok for tok in input_ids_ if tok in [fake_token_around_image_id, image_token_id]])
        if len_sample < max_seq_len:
            input_ids_ = input_ids_ + [pad_token_id] * (max_seq_len - len_sample)

        # Second we pad the image and pixel attention mask tensors
        # Max height and width of image accross individual samples
        if len(images_) > 0:
            max_height = max([im.size(1) for im in images_])
            max_width = max([im.size(2) for im in images_])
            padded_image_tensor = torch.zeros(max_num_images, 3, max_height, max_width)
            padded_pixel_attention_mask = torch.zeros(max_num_images, max_height, max_width, dtype=torch.bool)
            for idx, im in enumerate(images_):
                im_height, im_width = im.size()[1:]
                padded_image_tensor[idx, :, :im_height, :im_width] = im
                padded_pixel_attention_mask[idx, :im_height, :im_width] = True
            padded_image_tensor = padded_image_tensor.contiguous()
            padded_pixel_attention_mask = padded_pixel_attention_mask.contiguous()
        else:
            padded_image_tensor = None
            padded_pixel_attention_mask = None

        # Last we pad the input ids attention mask tensors
        attention_mask = torch.zeros((max_seq_len,), dtype=torch.long)
        attention_mask[:len_sample] = 1
        input_ids_ = torch.tensor(input_ids_)

        # Safety check to avoid batches with an unexpected amount of image_token_ids
        if (input_ids_ == image_token_id).sum() != image_seq_len * len(images_):
            logger.error(
                "Number of image_token_id should be the same as the number of images * image_seq_len. However, this"
                f" example: {input_ids_} has num_image_token_ids = {(input_ids_ == image_token_id).sum()} and images *"
                f" image_seq_len = {image_seq_len * len(images_)}. So we ignore it"
            )
            continue

        output_input_ids.append(input_ids_)
        output_num_text_tokens.append(len_sample - num_tokens_for_images)
        output_attention_masks.append(attention_mask)
        output_images.append(padded_image_tensor)
        output_pixel_attention_masks.append(padded_pixel_attention_mask)
        output_num_images.append(len(images_))

    if mask_labels:
        # Logic specific for sft tuning: we only compute the loss on the assistant part
        # That is a bit hacky workaround specifically for SFT.
        # The proper way would have been to handle the label masking inside each `split_pack_and_pad_X`, pass as input and output of `greedy_packing` and pass it as input and output of `prepare_return`
        # But that rather invasive change so doing that for now.
        if end_of_utterance_token_id is None:
            raise ValueError(
                "That logic has only been implemented at this point for computing the loss on the assistant answers in"
                " a user/assistant dialogue. We need `end_of_utterance_token_id`."
            )
        if bos_token_id is None or eos_token_id is None:
            raise ValueError(
                "Case where we don't separate packed sequence by `<BOS>` and `<EOS>` is not supported yet."
            )
        if assistant_token_ids is None:
            raise ValueError(
                "We were hoping to mask the part `\nAssistant:` too from the loss computation but"
                " `assistant_token_ids` is not specified."
            )

        def find_delimiters_tokens_to_mask(label_list):
            starts_ends_list = []
            start, end = None, None
            counter_eou = 0

            for idx, l_ in enumerate(label_list):
                if l_ == bos_token_id:
                    assert start is None and end is None, (idx, start, end)
                    start = idx
                elif l_ == end_of_utterance_token_id:
                    counter_eou += 1
                    if counter_eou % 2 != 0:
                        assert start is not None and end is None, (idx, start, end)
                        assert label_list[idx + 1 : idx + 1 + len(assistant_token_ids)] == assistant_token_ids
                        end = idx + 1 + len(assistant_token_ids)
                        starts_ends_list.append((start, end))
                        start, end = None, None
                    else:
                        assert start is None and end is None, (idx, start, end)
                        if idx + 1 < len(label_list) and label_list[idx + 1] != eos_token_id:
                            start = idx + 1
                elif l_ == eos_token_id:
                    assert start is None and end is None, (idx, start, end)
            assert start is None and end is None, (idx, start, end)

            return starts_ends_list

        output_labels = []
        for input_ids_ in output_input_ids:
            labels_ = input_ids_.clone()
            if (labels_ == end_of_utterance_token_id).sum() % 2 != 0:
                logger.error(
                    "Did not find an even number of `END_OF_UTTERANCE` tokens in the user/assistant dialogue. Not"
                    " masking the labels."
                )
                output_labels.append(labels_)
                continue

            starts_ends = find_delimiters_tokens_to_mask(labels_.tolist())
            for start_index, end_index in starts_ends:
                labels_[start_index:end_index] = image_token_id

            output_labels.append(labels_)
    else:
        output_labels = []

    return (
        output_input_ids,
        output_labels,
        output_images,
        output_attention_masks,
        output_pixel_attention_masks,
        output_num_images,
        output_num_text_tokens,
    )


def prepare_result_return(
    output_input_ids,
    output_images,
    output_attention_masks,
    output_pixel_attention_masks,
    output_num_images,
    output_num_text_tokens,
    output_labels=[],
):
    """
    This function returns the end dictionary at the exit of the dataloader.
    Mostly batchify things and pad accordingly.
    """
    if len(output_images) == 0 or len(output_input_ids) == 0:
        result = {
            "input_ids": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.bool),
            "pixel_attention_mask": torch.tensor([], dtype=torch.bool),
            "num_images": torch.tensor([], dtype=torch.long),
            "num_text_tokens": torch.tensor([], dtype=torch.long),
            "pixel_values": torch.tensor([], dtype=torch.float32),
        }
        return result

    output_input_ids = torch.stack(output_input_ids)
    output_attention_masks = torch.stack(output_attention_masks)

    total_batch_size = len(output_images)
    max_num_images = max([i.size(0) if i is not None else 0 for i in output_images])
    image_heights = [i.size(2) if i is not None else 0 for i in output_images]
    image_widths = [i.size(3) if i is not None else 0 for i in output_images]
    if max_num_images > 0:
        # Max height and width accross images in all packed samples
        max_height = max(image_heights)
        max_width = max(image_widths)
        padded_image_tensor = torch.zeros(total_batch_size, max_num_images, 3, max_height, max_width)
        padded_pixel_attention_masks = torch.zeros(
            total_batch_size, max_num_images, max_height, max_width, dtype=torch.bool
        )
        for idx, (sample_images, sample_pixel_attention_mask) in enumerate(
            zip(output_images, output_pixel_attention_masks)
        ):
            if sample_images is None:
                continue
            im_batch_height, im_batch_width = sample_images.size()[2:]
            padded_image_tensor[idx, :, :, :im_batch_height, :im_batch_width] = sample_images
            padded_pixel_attention_masks[idx, :, :im_batch_height, :im_batch_width] = sample_pixel_attention_mask
        padded_image_tensor = padded_image_tensor.contiguous()
        padded_pixel_attention_masks = padded_pixel_attention_masks.contiguous()
    else:
        padded_image_tensor = None
        padded_pixel_attention_masks = None

    # Sorting (and yielding to the dataloader) by text length + image sizes helps
    # reducing significantly the amount of padding (and thus wasted computed) when `shuffle_after_packing` is False.
    sort_by_padding = np.lexsort((output_attention_masks.sum(dim=-1).tolist(), image_heights, image_widths))

    result = {
        "input_ids": output_input_ids[sort_by_padding],
        "attention_mask": output_attention_masks[sort_by_padding],
        "num_images": torch.tensor(output_num_images)[sort_by_padding],
        "num_text_tokens": torch.tensor(output_num_text_tokens)[sort_by_padding],
    }
    if padded_pixel_attention_masks is not None:
        result["pixel_attention_mask"] = padded_pixel_attention_masks[sort_by_padding]
    if padded_image_tensor is not None:
        result["pixel_values"] = padded_image_tensor[sort_by_padding]

    if output_labels:
        output_labels = torch.stack(output_labels)
        result["labels"] = output_labels[sort_by_padding]
    return result


# Web documents
def split_pack_and_pad_webdocs(
    sample,
    tokenizer,
    max_seq_len,
    image_transform,
    max_num_images,
    image_seq_len,
    max_image_size=384,
    vision_encoder_max_image_size=384,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    max_num_samples_per_document=10,
    prefix_seed=(0, 0),
    add_begin_of_doc_token=True,
    add_end_of_doc_token=True,
    max_num_images_per_document=None,
    skip_ending_two_images=True,
    skip_multiple_consecutive_images=True,
):
    """
    Return a batch of samples in the format expected by the model which
    includes `input_ids`, `pixel_values`, `attention_mask`, `image_attention_mask`,
    and `next_image_attention_mask`. The `input_ids` are sampled from the document to
    ensure it has `max_seq_len` tokens otherwise, the shorter documents are packed together.
    For each document, we sample a maximum of `max_num_samples_per_document` or `max_num_samples_for_curr_document`
    (where the latter is proportional to the length of the document and inversely proportional to the length of subsequences)
    `input_ids` with sequence length `max_seq_len` from the document. This means that
    each sample sampled can have different start index. Based on the start index of sample that
    has been sampled, we also sample a maximum of `max_num_images` images from the document.
    If there are less than `max_num_images` images in the document, we pad the images with zeros.
    The start indexes are skewed towards subsequences that contain images.

    Args:
        sample (Dict): A sample object containing the document with images and text.
        tokenizer (PretrainedTokenizer): Text tokenizer to be used.
        max_seq_len (int): Maximum sequence length of the returned text tokens.
        image_transform (Callable): Transform to be applied on the images
        max_num_images (int): Maximum number of images to be sampled per sample. If less, they are padded with zeros.
        max_num_samples_per_document (int, optional): Maximum number of samples per document to be sampled. Defaults to 10.
        prefix_seed: Prefix seed sequence for "reproducible randomness" in calls to `np.random.choice`

    Returns:
        _type_: _description_
    """
    text_batch = sample["texts"]

    image_batch = sample.get("images", None)
    if image_batch is None:
        raise ValueError("`images` must be present in the sample")

    pad_token_id = tokenizer.pad_token_id

    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    last_was_image = False

    fake_token_around_image_id = tokenizer.convert_tokens_to_ids(FAKE_TOKEN_AROUND_IMAGE_V2)
    MAX_NUM_IMAGE_ROWS_AND_COLUMNS = math.ceil(max_image_size / vision_encoder_max_image_size)
    MAX_NUM_IMAGES_AFTER_SPLIT = (
        MAX_NUM_IMAGE_ROWS_AND_COLUMNS**2 + 1 * (MAX_NUM_IMAGE_ROWS_AND_COLUMNS != 1)
    ) * max_num_images

    # We need to encode the \n\n with another token that we remove afterwards to prevent the tokenizer from generating
    # the prefix token.
    double_breaking_lines_token_ids = tokenizer.encode("_\n\n", add_special_tokens=False)
    double_breaking_lines_token_ids = [
        tok for tok in double_breaking_lines_token_ids if tok not in tokenizer.encode("_", add_special_tokens=False)
    ]

    all_images = []
    all_texts = []
    for raw_images, raw_texts in zip(image_batch, text_batch):
        # Filter ones that don't have either one image and one text word
        if not any(raw_images) or not any(raw_texts):
            continue

        if max_num_images_per_document:
            num_images = sum([1 if image is not None else 0 for image in raw_images])
            if num_images > max_num_images_per_document:
                continue

        if skip_ending_two_images:
            # Skipping sequences that end with a concatenation of at least two images with no text in between
            # Order of magnitude: skipping 10% of sequences
            if (len(raw_texts) >= 2) and (raw_texts[-1] is None) and (raw_texts[-2] is None):
                continue

        def has_consecutive_nones(lst, max_num_nones=3):
            count = 0
            for item in lst:
                if item is None:
                    count += 1
                    if count == max_num_nones:
                        return True
                else:
                    count = 0
            return False

        if skip_multiple_consecutive_images and has_consecutive_nones(raw_texts):
            # We skip documents with at least 3 consecutive images
            continue

        inds_of_texts_to_split = [
            i
            for i, text in enumerate(raw_texts)
            if text is not None and isinstance(text, str) and "END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED" in text
        ]
        if inds_of_texts_to_split:
            splitted_raw_images, splitted_raw_texts = [], []
            previous_i = 0
            for i in inds_of_texts_to_split:
                splitting = raw_texts[i].split("END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED")
                part1, part2 = splitting[0], splitting[-1]

                sub_doc_images = raw_images[previous_i:i] + [None]
                sub_doc_texts = raw_texts[previous_i:i] + [part1.strip()]
                if not any(sub_doc_images):  # This can happen if all images in raw_images[0:i] are all None
                    continue

                splitted_raw_images.append(sub_doc_images)
                splitted_raw_texts.append(sub_doc_texts)

                if part2.strip() == "":
                    previous_i = i + 1
                else:
                    raw_texts[i] = part2.strip()
                    previous_i = i

            if previous_i < len(raw_images) and any(raw_images[previous_i:]):
                splitted_raw_images.append(raw_images[previous_i:])
                splitted_raw_texts.append(raw_texts[previous_i:])

        else:
            splitted_raw_images, splitted_raw_texts = [raw_images], [raw_texts]

        # Sanity check
        if [len(ims) for ims in splitted_raw_images] != [len(txts) for txts in splitted_raw_texts]:
            raise ValueError(
                "Number of images and texts don't match after splitting on `END_OF_DOCUMENT_TOKEN_TO_BE_REPLACED`."
                " Something core went wrong during the splitting and needs to be fixed."
            )

        for s_r_ims, s_r_txts in zip(splitted_raw_images, splitted_raw_texts):
            images, web_text = [], ""
            for image, text in zip(s_r_ims, s_r_txts):
                if text is None and image is None:
                    continue
                if (text is not None) and ((FAKE_TOKEN_AROUND_IMAGE_V2 in text) or (IMAGE_TOKEN in text)):
                    continue

                if image is not None:
                    splitted_image_array, text_splitted_images = get_splitted_images_and_corresponding_text(
                        image=image,
                        vision_encoder_max_image_size=vision_encoder_max_image_size,
                        max_image_size=max_image_size,
                        pre_split_scale_up_max=pre_split_scale_up_max,
                        pre_split_scale_up_frequency=pre_split_scale_up_frequency,
                        image_seq_len=image_seq_len,
                    )
                    web_text += text_splitted_images
                    images.extend([image_transform(img) for img in splitted_image_array])
                    last_was_image = True
                elif text is not None:
                    if last_was_image:
                        web_text += f"{text}"
                        last_was_image = False
                    else:
                        web_text += f" {text}" if web_text != "" else text

            web_text = web_text.strip(" ")

            # This is mostly a sanity check. Cases like that should not happen at that point.
            if web_text == "" or len(images) == 0:
                continue

            all_images.append(images)

            web_text_ids = tokenizer.encode(web_text, add_special_tokens=False)
            if add_end_of_doc_token:
                web_text_ids += [tokenizer.eos_token_id]

            if add_begin_of_doc_token:
                web_text_ids = [tokenizer.bos_token_id] + web_text_ids
            all_texts.append(web_text_ids)

    output_input_ids = []
    output_images = []
    output_attention_masks = []
    output_pixel_attention_masks = []
    output_num_images = []
    output_num_text_tokens = []

    input_ids_to_pack = []
    images_to_pack = []
    for images, text in zip(all_images, all_texts):
        # We shouldn't train on documents containing only one image that is at the end
        # because no text token could attend that image.
        # -2 is because we can have the eos token at position -1
        if (len(images) == 1) and (
            (text[-2] == fake_token_around_image_id) or (text[-1] == fake_token_around_image_id)
        ):
            continue

        # We save all the documents which are shorter than the max_seq_len to pack them together.
        if len(text) <= max_seq_len:
            if len(text) < _MIN_LENGTH_DOCUMENTS_TO_PACK:  # Filter out extremely short sequences
                continue
            input_ids_to_pack.append(text)
            images_to_pack.append(images)
        else:
            # We disable the following logic for the case of image splitting.
            # When doing image splitting, with the current implementation, we could have an image
            # that is incomplete. Moreover, most documents from Obelics contain less than 1024 tokens.
            # It can make sense to skip them for now and re-implement a better method later on.
            continue

            # Computing the bonus scores for tokens near images to skew the sampling towards them
            # The main idea is to give a bonus to tokens that are closely before an image token, so that these tokens have more chance to be sampled.
            # Bonuses are computed for each image, which means a given token can receive bonuses from multiple images if this token is closely preceding multiple images.
            # We sum all the bonuses and L1 normalized along the seq_len axis to get a probability distribution.
            # Each token start with a regular bonus of 1, which corresponds to the uniform distribution over the sequence when there are no bonuses added.
            all_scores = np.array([1] * len(text))
            starting_pos_images = [
                idx
                for idx in range(1, len(text))
                if text[idx] == image_token_id and text[idx - 1] == fake_token_around_image_id
            ]
            for img_token_idx in starting_pos_images:
                if (
                    image_seq_len >= max_seq_len
                ):  # This case shouldn't happen in any case, because we wouldn't fit any images otherwise
                    raise ValueError(
                        f"image_seq_len ({image_seq_len}) should be shorter than max_seq_len ({max_seq_len})"
                    )
                # We don't want to give a bonus to text tokens after the image, otherwise we would start
                # from there and not see the image, so we have to stop at least at img_token_idx on the right.
                # We don't want to start with a text token before img_token_idx - 1 + image_seq_len - max_seq_len,
                # before, otherwise the image would be truncated.
                # Even if the image is not truncated, it is useless if there isn't any text tokens after the image,
                # so we start a bit after on the left, at img_token_idx - 1 + image_seq_len - 0.75*max_seq_len, such
                # that at least 25% of the text tokens are after the image.
                all_scores[
                    max(0, img_token_idx - 1 + image_seq_len - int(0.75 * max_seq_len)) : img_token_idx
                ] += _IMAGE_BONUS_VALUE
            # Penalty in order not to start in the middle of an image sequence
            for img_token_idx in starting_pos_images:
                all_scores[img_token_idx : img_token_idx + image_seq_len] = 0
                # Unless the next token after the fake token is a new image, we should score it at 0 as well
                end_fake_token_idx = img_token_idx + image_seq_len
                if end_fake_token_idx + 1 >= len(text):
                    all_scores[end_fake_token_idx] = 0
                elif text[end_fake_token_idx + 1] != image_token_id:
                    all_scores[end_fake_token_idx] = 0
            all_scores = all_scores[:-_MIN_LENGTH_DOCUMENTS_TO_PACK]

            # The number of samples is proportional to the length of the text and inversely proportional to the maximum sequence length
            max_num_samples_for_curr_document = len(text) // max_seq_len
            # Set "reproducible randomness" by creating an np.default_rng seeded by (main seed, epoch, rank_idx, worker_idx, mapped_batch_index, text len)
            choices = np.random.default_rng(seed=list(prefix_seed) + [len(text)]).choice(
                range(len(text) - _MIN_LENGTH_DOCUMENTS_TO_PACK),  # shorter sub-sequences are reserved for packing
                min(
                    len(text) - max_seq_len, 2 * max_num_samples_per_document
                ),  # Sampling more than necessary and then breaking out of the for loop once we have enough samples
                p=all_scores / np.linalg.norm(all_scores, ord=1),
                replace=False,
            )

            nb_effective_sequences_out_of_sampling = 0
            for start_index in choices:
                # We should start at start_index. However, there are particular cases.
                text_sub_sequence = text[start_index:]

                # First case, we start at the fake token ending an image
                if len(text_sub_sequence) > 1:
                    if (text_sub_sequence[0] == fake_token_around_image_id) and (
                        text_sub_sequence[1] != image_token_id
                    ):
                        text_sub_sequence = text_sub_sequence[1:]
                        start_index += 1
                else:
                    continue

                # Second case, we are in the middle of the image sequence, we skip the first image
                if text_sub_sequence[0] == image_token_id:
                    try:
                        first_occurrence_fake_token = text_sub_sequence.index(fake_token_around_image_id)
                        text_sub_sequence = text_sub_sequence[first_occurrence_fake_token:]
                        start_index += first_occurrence_fake_token
                        if len(text_sub_sequence) > 1:
                            if text_sub_sequence[1] != image_token_id:
                                text_sub_sequence = text_sub_sequence[1:]
                                start_index += 1
                        else:
                            continue
                    except IndexError:
                        # Case where `fake_token_around_image_id` is not found in `text_sub_sequence`
                        # We don't have complete images in the example so we skip it
                        continue
                    else:
                        raise ValueError("something else went wrong")

                # Now that we finished the truncation on the left, we can truncate on the right
                if (text_sub_sequence[0] != tokenizer.bos_token_id) and add_begin_of_doc_token:
                    text_sub_sequence = [tokenizer.bos_token_id] + text_sub_sequence
                # We don't add eos tokens to truncated documents, but it's important in this case to add the bos
                # token, otherwise two documents can be packed together without any boundary.
                text_sub_sequence = text_sub_sequence[:max_seq_len]

                # Third case, we finish in the middle of the image sequence, so we skip it
                # Everything is autoregressive and we are not computing the loss on the image tokens, so we
                # could end with an incomplete image. However, it makes things simpler for counting the images
                # or replacing the image sequence in the modeling to simply discard it. Moreover, if the resulting
                # `text_sub_sequence` is shorter than `max_seq_len`, we use the example in the packing,
                # so we could have an incomplete image right in the middle of the packed sequence
                if text_sub_sequence[-1] == fake_token_around_image_id:
                    if text_sub_sequence[-2] != image_token_id:
                        text_sub_sequence = text_sub_sequence[:-1]
                elif text_sub_sequence[-1] == image_token_id:
                    try:
                        last_occurrence_fake_token_idx = (
                            len(text_sub_sequence) - 1 - text_sub_sequence[::-1].index(fake_token_around_image_id)
                        )
                        text_sub_sequence = text_sub_sequence[:last_occurrence_fake_token_idx]
                    except Exception:
                        continue

                # Check that we still have a sufficiently long sequence
                if len(text) < _MIN_LENGTH_DOCUMENTS_TO_PACK:
                    continue

                num_image_tokens = text_sub_sequence.count(image_token_id)
                image_count = num_image_tokens // image_seq_len
                if image_count == 0:
                    # Skip if there are no images in the sequence
                    continue
                # We shouldn't train on documents containing only one image that is at the end
                # because no text token could attend that image.
                if (image_count == 1) and (text_sub_sequence[-1] == fake_token_around_image_id):
                    continue

                image_start_index = len([start_pos for start_pos in starting_pos_images if start_pos < start_index])
                current_images = images[image_start_index : image_start_index + image_count]

                num_images_to_discard = len(current_images) - MAX_NUM_IMAGES_AFTER_SPLIT
                if num_images_to_discard > 0:
                    text_sub_sequence, current_images = remove_extra_images(
                        input_ids_=text_sub_sequence,
                        images_=current_images,
                        max_num_images=MAX_NUM_IMAGES_AFTER_SPLIT,
                        image_seq_len=image_seq_len,
                        fake_token_around_image_id=fake_token_around_image_id,
                        image_token_id=image_token_id,
                        double_breaking_lines_token_ids=double_breaking_lines_token_ids,
                    )

                if len(text_sub_sequence) < max_seq_len:
                    # If the sub-sequence is shorter than max_seq_len, we reserve it for packing
                    input_ids_to_pack.append(text_sub_sequence)
                    images_to_pack.append(current_images)

                    nb_effective_sequences_out_of_sampling += 1
                    if nb_effective_sequences_out_of_sampling >= min(
                        max_num_samples_for_curr_document, max_num_samples_per_document
                    ):
                        # We got all the samples we need for this document, so breaking out
                        break
                    else:
                        continue

                text_sub_sequence = torch.tensor(text_sub_sequence)
                # Safety check to avoid batches with an unexpected amount of image_token_ids
                if (text_sub_sequence == image_token_id).sum() != image_seq_len * len(current_images):
                    logger.error(
                        "Number of image_token_id should be the same as the number of images * image_seq_len."
                        f" However, this example: {text_sub_sequence} has num_image_token_ids ="
                        f" {(text_sub_sequence == image_token_id).sum()} and images * image_seq_len ="
                        f" {image_seq_len * len(current_images)}. So we ignore it"
                    )
                    continue

                max_height = max([im.size(1) for im in current_images])
                max_width = max([im.size(2) for im in current_images])
                padded_image_tensor = torch.zeros(MAX_NUM_IMAGES_AFTER_SPLIT, 3, max_height, max_width)
                padded_pixel_attention_mask = torch.zeros(
                    MAX_NUM_IMAGES_AFTER_SPLIT, max_height, max_width, dtype=torch.bool
                )
                for idx, im in enumerate(current_images):
                    im_height, im_width = im.size()[1:]
                    padded_image_tensor[idx, :, :im_height, :im_width] = im
                    padded_pixel_attention_mask[idx, :im_height, :im_width] = True
                padded_image_tensor = padded_image_tensor.contiguous()
                padded_pixel_attention_mask = padded_pixel_attention_mask.contiguous()

                output_images.append(padded_image_tensor)
                output_pixel_attention_masks.append(padded_pixel_attention_mask)
                output_num_images.append(min(MAX_NUM_IMAGES_AFTER_SPLIT, image_count))

                output_input_ids.append(text_sub_sequence)
                output_num_text_tokens.append(len(text_sub_sequence))

                attention_mask = torch.ones((max_seq_len,), dtype=torch.long)
                output_attention_masks.append(attention_mask)

                nb_effective_sequences_out_of_sampling += 1
                if nb_effective_sequences_out_of_sampling >= min(
                    max_num_samples_for_curr_document, max_num_samples_per_document
                ):
                    # We got all the samples we need for this document, so breaking out
                    break

    # Pack the remaining sequences from `input_ids_to_pack` x `images_to_pack`
    if input_ids_to_pack:
        (
            output_input_ids,
            _,
            output_images,
            output_attention_masks,
            output_pixel_attention_masks,
            output_num_images,
            output_num_text_tokens,
        ) = greedy_packing(
            input_ids_to_pack=input_ids_to_pack,
            images_to_pack=images_to_pack,
            max_seq_len=max_seq_len,
            max_num_images=MAX_NUM_IMAGES_AFTER_SPLIT,
            image_seq_len=image_seq_len,
            pad_token_id=pad_token_id,
            fake_token_around_image_id=fake_token_around_image_id,
            image_token_id=image_token_id,
            double_breaking_lines_token_ids=double_breaking_lines_token_ids,
            output_input_ids=output_input_ids,
            output_images=output_images,
            output_attention_masks=output_attention_masks,
            output_pixel_attention_masks=output_pixel_attention_masks,
            output_num_images=output_num_images,
            output_num_text_tokens=output_num_text_tokens,
            truncate_images_within_same_example=False,
        )

    result = prepare_result_return(
        output_input_ids=output_input_ids,
        output_images=output_images,
        output_attention_masks=output_attention_masks,
        output_pixel_attention_masks=output_pixel_attention_masks,
        output_num_images=output_num_images,
        output_num_text_tokens=output_num_text_tokens,
    )
    return result


# Image text pairs
def split_pack_and_pad_pairs(
    sample,
    tokenizer,
    max_seq_len,
    image_transform,
    max_num_images,
    image_seq_len,
    max_image_size=384,
    vision_encoder_max_image_size=384,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    prefix_seed=(0, 0),
    add_begin_of_doc_token=True,
    add_end_of_doc_token=True,
):
    pad_token_id = tokenizer.pad_token_id
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    fake_token_around_image_id = tokenizer.convert_tokens_to_ids(FAKE_TOKEN_AROUND_IMAGE_V2)
    # We need to encode the \n\n with another token that we remove afterwards to prevent the tokenizer from generating
    # the prefix token.
    double_breaking_lines_token_ids = tokenizer.encode("_\n\n", add_special_tokens=False)
    double_breaking_lines_token_ids = [
        tok for tok in double_breaking_lines_token_ids if tok not in tokenizer.encode("_", add_special_tokens=False)
    ]
    MAX_NUM_IMAGE_ROWS_AND_COLUMNS = math.ceil(max_image_size / vision_encoder_max_image_size)
    MAX_NUM_IMAGES_AFTER_SPLIT = (
        MAX_NUM_IMAGE_ROWS_AND_COLUMNS**2 + 1 * (MAX_NUM_IMAGE_ROWS_AND_COLUMNS != 1)
    ) * max_num_images

    text_batch = sample["text"]
    image_batch = sample.get("image", None)
    if image_batch is None:
        raise ValueError("`images` must be present in the sample")

    filtered_image_batch = []
    filtered_input_ids = []

    for image, text in zip(image_batch, text_batch):
        if text is None or image is None:
            continue
        if (text is not None) and ((FAKE_TOKEN_AROUND_IMAGE_V2 in text) or (IMAGE_TOKEN in text)):
            continue

        splitted_image_array, sample_text = get_splitted_images_and_corresponding_text(
            image=image,
            vision_encoder_max_image_size=vision_encoder_max_image_size,
            max_image_size=max_image_size,
            pre_split_scale_up_max=pre_split_scale_up_max,
            pre_split_scale_up_frequency=pre_split_scale_up_frequency,
            image_seq_len=image_seq_len,
        )

        # Remove trailing and leading whitespaces, including newlines and tabs
        text = text.strip()

        sample_text = f"{sample_text}{text}"

        sample_input_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        if add_end_of_doc_token:
            sample_input_ids += [tokenizer.eos_token_id]

        if add_begin_of_doc_token:
            sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids

        if len(sample_input_ids) > max_seq_len:
            continue

        filtered_image_batch.append([image_transform(image) for image in splitted_image_array])

        filtered_input_ids.append(sample_input_ids)

    input_ids_to_pack = filtered_input_ids
    images_to_pack = filtered_image_batch

    (
        output_input_ids,
        _,
        output_images,
        output_attention_masks,
        output_pixel_attention_masks,
        output_num_images,
        output_num_text_tokens,
    ) = greedy_packing(
        input_ids_to_pack=input_ids_to_pack,
        images_to_pack=images_to_pack,
        max_seq_len=max_seq_len,
        max_num_images=MAX_NUM_IMAGES_AFTER_SPLIT,
        image_seq_len=image_seq_len,
        pad_token_id=pad_token_id,
        fake_token_around_image_id=fake_token_around_image_id,
        image_token_id=image_token_id,
        double_breaking_lines_token_ids=double_breaking_lines_token_ids,
        output_input_ids=[],
        output_images=[],
        output_attention_masks=[],
        output_pixel_attention_masks=[],
        output_num_images=[],
        output_num_text_tokens=[],
        truncate_images_within_same_example=False,
    )
    result = prepare_result_return(
        output_input_ids=output_input_ids,
        output_images=output_images,
        output_attention_masks=output_attention_masks,
        output_pixel_attention_masks=output_pixel_attention_masks,
        output_num_images=output_num_images,
        output_num_text_tokens=output_num_text_tokens,
    )
    return result


# Ocr data
def split_pack_and_pad_ocr(
    sample,
    tokenizer,
    max_seq_len,
    image_transform,
    max_num_images,
    image_seq_len,
    max_image_size=384,
    vision_encoder_max_image_size=384,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    prefix_seed=(0, 0),
    add_begin_of_doc_token=True,
    add_end_of_doc_token=True,
):
    pad_token_id = tokenizer.pad_token_id
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    fake_token_around_image_id = tokenizer.convert_tokens_to_ids(FAKE_TOKEN_AROUND_IMAGE_V2)
    # We need to encode the \n\n with another token that we remove afterwards to prevent the tokenizer from generating
    # the prefix token.
    double_breaking_lines_token_ids = tokenizer.encode("_\n\n", add_special_tokens=False)
    double_breaking_lines_token_ids = [
        tok for tok in double_breaking_lines_token_ids if tok not in tokenizer.encode("_", add_special_tokens=False)
    ]

    # Make sure the seq_len of the max_num_image_input_ids is inferior to max_seq_len
    MAX_NUM_IMAGE_ROWS_AND_COLUMNS = math.ceil(max_image_size / vision_encoder_max_image_size)
    MAX_IMAGE_TEXT_SEQUENCE = (
        ((f"{FAKE_TOKEN_AROUND_IMAGE_V2}" + f"{IMAGE_TOKEN}" * image_seq_len) * MAX_NUM_IMAGE_ROWS_AND_COLUMNS + "\n")
        * MAX_NUM_IMAGE_ROWS_AND_COLUMNS
        + f"\n{FAKE_TOKEN_AROUND_IMAGE_V2}"
        + f"{IMAGE_TOKEN}" * image_seq_len
        + f"{FAKE_TOKEN_AROUND_IMAGE_V2}"
    )
    max_image_input_ids_seq = tokenizer.encode(MAX_IMAGE_TEXT_SEQUENCE, add_special_tokens=False)
    len_max_num_image_input_ids = len(max_image_input_ids_seq) * max_num_images
    if len_max_num_image_input_ids >= max_seq_len:
        raise ValueError(
            f"max_image_seq_len ({len_max_num_image_input_ids}) should be shorter than max_seq_len ({max_seq_len}). with image_seq_len ({image_seq_len})"
        )
    # The max_num_images_after split given the max_image size, vision_encoder_max_image_size and max_num_images
    MAX_NUM_IMAGES_AFTER_SPLIT = (
        MAX_NUM_IMAGE_ROWS_AND_COLUMNS**2 + 1 * (MAX_NUM_IMAGE_ROWS_AND_COLUMNS != 1)
    ) * max_num_images

    text_batch = sample["texts"]
    image_batch = sample.get("images", None)
    if image_batch is None:
        raise ValueError("`images` must be present in the sample")

    def tokenize_text_sublist(curr_text_sublist, curr_image_sublist_text, curr_image_text, tokenizer):
        sample_text = "\n\n".join(curr_text_sublist).strip()
        sample_text = f"{curr_image_sublist_text}{curr_image_text}{sample_text}"

        sample_input_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        if add_end_of_doc_token:
            sample_input_ids += [tokenizer.eos_token_id]

        if add_begin_of_doc_token:
            sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids
        return sample_input_ids

    filtered_image_batch = []
    filtered_input_ids = []
    # Iterate through the lists of images and texts. Each list tuple contains a full pdf document with all the images and annotations associated
    for image_list, text_list in zip(image_batch, text_batch):
        len_text_list = len(text_list)
        curr_image_sublist = []
        curr_text_sublist = []
        curr_image_sublist_text = ""
        for i, (curr_image, curr_text) in enumerate(zip(image_list, text_list)):
            curr_text_sublist.append(curr_text)

            splitted_image_array, curr_image_text = get_splitted_images_and_corresponding_text(
                image=curr_image,
                vision_encoder_max_image_size=vision_encoder_max_image_size,
                max_image_size=max_image_size,
                pre_split_scale_up_max=pre_split_scale_up_max,
                pre_split_scale_up_frequency=pre_split_scale_up_frequency,
                image_seq_len=image_seq_len,
            )

            sample_input_ids = tokenize_text_sublist(
                curr_text_sublist=curr_text_sublist,
                curr_image_sublist_text=curr_image_sublist_text,
                curr_image_text=curr_image_text,
                tokenizer=tokenizer,
            )

            # If a sublist is longer than 1 element, but its encoded ids are longer than the max_seq len, we create an example with all but the last element of the curr_text_sublist
            # (it was kept at the previous iteration, so the len(sample_input_ids) of curr_text_sublist[:-1] has to be < max_seq_len). Then we keep the last element of the
            # curr_text_sublist and create a curr_image_sublist with the latest image to keep for the next iteration.
            if (
                len(sample_input_ids) > max_seq_len
                or len(curr_image_sublist) + len(splitted_image_array) > MAX_NUM_IMAGES_AFTER_SPLIT
            ) and len(curr_text_sublist) > 1:
                future_text_sublist = [curr_text_sublist[-1]]
                curr_text_sublist = curr_text_sublist[:-1]
                sample_input_ids = tokenize_text_sublist(
                    curr_text_sublist=curr_text_sublist,
                    curr_image_sublist_text=curr_image_sublist_text,
                    curr_image_text="",
                    tokenizer=tokenizer,
                )
                filtered_image_batch.append([image_transform(image) for image in curr_image_sublist])
                filtered_input_ids.append(sample_input_ids)

                # Current text sublist is now only the last element of the previous sublist. We tokenize the new text sublist as we now need to check if it is longer than max_seq_len,
                # otherwise it will be added to the next iteration sublist
                curr_text_sublist = future_text_sublist
                sample_input_ids = tokenize_text_sublist(
                    curr_text_sublist=curr_text_sublist,
                    curr_image_sublist_text="",
                    curr_image_text=curr_image_text,
                    tokenizer=tokenizer,
                )
                curr_image_sublist = []
                curr_image_sublist_text = ""

            # If a sublist of only 1 is longer than the sequence length, we create multiple examples with the same image, but text corresponding to different parts of the doc
            if len(sample_input_ids) > max_seq_len and len(curr_text_sublist) == 1:
                list_sample_input_ids = [sample_input_ids[:max_seq_len]]
                image_input_ids_seq = tokenizer.encode(curr_image_text, add_special_tokens=False)
                max_len_input_ids_chunk = max_seq_len - len(image_input_ids_seq)
                for chunk_start_index in range(max_seq_len, len(sample_input_ids), max_len_input_ids_chunk):
                    list_sample_input_ids.append(
                        image_input_ids_seq
                        + sample_input_ids[chunk_start_index : chunk_start_index + max_len_input_ids_chunk]
                    )
                for sample in list_sample_input_ids:
                    filtered_image_batch.append([image_transform(image) for image in splitted_image_array])
                    filtered_input_ids.append(sample)

                # reset the sublists for the next iteration
                curr_image_sublist = []
                curr_text_sublist = []
                curr_image_sublist_text = ""
            # If len(sample_input_ids) < max_seq_len, we add the new image to the curr_image_sublist and either try to increase the length of the sublists further,
            # or add the example if this is the end of the doc text_list or if we passed the MAX_NUM_IMAGES_AFTER_SPLIT (it can be a bit more if the image is split
            else:
                curr_image_sublist.extend(splitted_image_array)
                curr_image_sublist_text += curr_image_text
                if i + 1 == len_text_list or len(curr_image_sublist) == MAX_NUM_IMAGES_AFTER_SPLIT:
                    filtered_image_batch.append([image_transform(image) for image in curr_image_sublist])
                    filtered_input_ids.append(sample_input_ids)
                    curr_image_sublist = []
                    curr_text_sublist = []
                    curr_image_sublist_text = ""

    (
        output_input_ids,
        _,
        output_images,
        output_attention_masks,
        output_pixel_attention_masks,
        output_num_images,
        output_num_text_tokens,
    ) = greedy_packing(
        input_ids_to_pack=filtered_input_ids,
        images_to_pack=filtered_image_batch,
        max_seq_len=max_seq_len,
        max_num_images=MAX_NUM_IMAGES_AFTER_SPLIT,
        image_seq_len=image_seq_len,
        pad_token_id=pad_token_id,
        fake_token_around_image_id=fake_token_around_image_id,
        image_token_id=image_token_id,
        double_breaking_lines_token_ids=double_breaking_lines_token_ids,
        output_input_ids=[],
        output_images=[],
        output_attention_masks=[],
        output_pixel_attention_masks=[],
        output_num_images=[],
        output_num_text_tokens=[],
        truncate_images_within_same_example=False,
    )
    result = prepare_result_return(
        output_input_ids=output_input_ids,
        output_images=output_images,
        output_attention_masks=output_attention_masks,
        output_pixel_attention_masks=output_pixel_attention_masks,
        output_num_images=output_num_images,
        output_num_text_tokens=output_num_text_tokens,
    )
    return result


# Image/Question/Answer triplets
def split_pack_and_pad_iqa_finetuning(
    sample,
    tokenizer,
    max_seq_len,
    image_transform,
    max_num_images,
    image_seq_len,
    prefix_seed=(0, 0),
    add_begin_of_doc_token=True,
    add_end_of_doc_token=True,
):
    question_batch = sample["question"]
    answer_batch = sample["answer"]
    image_batch = sample.get("image", None)
    if image_batch is None:
        raise ValueError("`image` must be present in the sample")

    filtered_image_batch = []
    filtered_input_ids = []

    for image, question, answer in zip(image_batch, question_batch, answer_batch):
        if question is None or answer is None or image is None:
            continue

        sample_text = (
            f"{FAKE_TOKEN_AROUND_IMAGE_V2}" + f"{IMAGE_TOKEN}" * image_seq_len + f"{FAKE_TOKEN_AROUND_IMAGE_V2}"
        )

        # Remove trailing and leading whitespaces, including newlines and tabs
        question = question.strip()
        answer = answer.strip()

        sample_text = f"{sample_text}Question: {question}\nAnswer: {answer}"

        sample_input_ids = tokenizer.encode(sample_text, add_special_tokens=False)
        sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids + [tokenizer.eos_token_id]

        if len(sample_input_ids) > max_seq_len:
            continue

        filtered_image_batch.append(image)
        filtered_input_ids.append(sample_input_ids)

    all_images = []
    all_texts = []
    all_attention_masks = []
    all_num_images = []
    all_num_text_tokens = []

    for image, sample_input_ids in zip(filtered_image_batch, filtered_input_ids):
        current_images = [image_transform(image)]
        current_images = torch.stack(current_images)
        all_num_images.append(1)
        all_images.append(current_images)

        padded_input_ids = torch.full((max_seq_len,), tokenizer.pad_token_id)
        current_max_len = min(max_seq_len, len(sample_input_ids))
        padded_input_ids[:current_max_len] = torch.tensor(sample_input_ids)[:current_max_len]
        all_num_text_tokens.append(current_max_len)
        all_texts.append(padded_input_ids)

        attention_mask = torch.zeros((max_seq_len,), dtype=torch.long)
        attention_mask[: len(sample_input_ids)] = 1
        all_attention_masks.append(attention_mask)

    if len(all_images) == 0 or len(all_texts) == 0:
        result = {
            "input_ids": torch.tensor([], dtype=torch.long),
            "attention_mask": torch.tensor([], dtype=torch.bool),
            "num_images": torch.tensor([], dtype=torch.long),
            "num_text_tokens": torch.tensor([], dtype=torch.long),
            "pixel_values": torch.tensor([], dtype=torch.float32),
        }
        return result

    all_texts = torch.stack(all_texts)
    all_images = torch.stack(all_images)
    all_attention_masks = torch.stack(all_attention_masks)

    output = {
        "input_ids": all_texts,
        "attention_mask": all_attention_masks,
        "num_images": torch.tensor(all_num_images),
        "num_text_tokens": torch.tensor(all_num_text_tokens),
        "pixel_values": all_images,
    }

    return output


# Sft
def split_pack_and_pad_sft(
    sample,
    tokenizer,
    max_seq_len,
    image_transform,
    max_num_images,
    image_seq_len,
    max_image_size=384,
    vision_encoder_max_image_size=384,
    pre_split_scale_up_max=1.0,
    pre_split_scale_up_frequency=0.0,
    prefix_seed=(0, 0),
    add_begin_of_doc_token=True,
    add_end_of_doc_token=True,
):
    MAX_NUMBER_OF_TURNS = 7
    LIST_OF_SFT_DATASETS_WITH_TURNS_ORDER = ["ny_cc_ranking"]
    pad_token_id = tokenizer.pad_token_id
    image_token_id = tokenizer.convert_tokens_to_ids(IMAGE_TOKEN)
    fake_token_around_image_id = tokenizer.convert_tokens_to_ids(FAKE_TOKEN_AROUND_IMAGE_V2)
    MAX_NUM_IMAGE_ROWS_AND_COLUMNS = math.ceil(max_image_size / vision_encoder_max_image_size)
    MAX_NUM_IMAGES_AFTER_SPLIT = (
        MAX_NUM_IMAGE_ROWS_AND_COLUMNS**2 + 1 * (MAX_NUM_IMAGE_ROWS_AND_COLUMNS != 1)
    ) * max_num_images
    # We need to encode the \n\n with another token that we remove afterwards to prevent the tokenizer from generating
    # the prefix token.
    double_breaking_lines_token_ids = tokenizer.encode("_\n\n", add_special_tokens=False)
    double_breaking_lines_token_ids = [
        tok for tok in double_breaking_lines_token_ids if tok not in tokenizer.encode("_", add_special_tokens=False)
    ]
    end_of_utterance_token_id = tokenizer.convert_tokens_to_ids(END_OF_UTTERANCE_TOKEN)
    assistant_token_ids = tokenizer("\nAssistant:", add_special_tokens=False)["input_ids"]

    text_batch = sample["texts"]
    image_batch = sample.get("images", None)
    if image_batch is None:
        raise ValueError("`images` must be present in the sample")

    filtered_image_batch = []
    filtered_input_ids = []

    for images, turns in zip(image_batch, text_batch):
        if not turns:
            continue
        if any([(FAKE_TOKEN_AROUND_IMAGE_V2 in t["user"] or IMAGE_TOKEN in t["user"]) for t in turns]):
            continue
        if any([(FAKE_TOKEN_AROUND_IMAGE_V2 in t["assistant"] or IMAGE_TOKEN in t["assistant"]) for t in turns]):
            continue
        if "PDFA key" in turns[0]["source"]:
            # Quick and dirty filtering for Large DocVQA
            if any([(len(t["user"]) > 500) or (len(t["assistant"]) > 500) for t in turns]):
                continue

        images_text = ""
        for idx_image, image in enumerate(images):
            images[idx_image], text_splitted_images = get_splitted_images_and_corresponding_text(
                image=image,
                vision_encoder_max_image_size=vision_encoder_max_image_size,
                max_image_size=max_image_size,
                pre_split_scale_up_max=pre_split_scale_up_max,
                pre_split_scale_up_frequency=pre_split_scale_up_frequency,
                image_seq_len=image_seq_len,
            )
            images_text += text_splitted_images

        images = [sub_el for el in images for sub_el in el]

        text = ""
        if turns[0]["source"] not in LIST_OF_SFT_DATASETS_WITH_TURNS_ORDER:
            random.shuffle(turns)
        for idx, t in enumerate(turns[:MAX_NUMBER_OF_TURNS]):
            random_linebreak = "\n" if random.random() < RANDOM_LINE_BREAK_PROB else ""
            user_text = t["user"].strip()
            assistant_text = t["assistant"].strip()
            if idx == 0:
                text += (
                    f"User:{images_text if images_text else ' '}{random_linebreak}{user_text}{END_OF_UTTERANCE_TOKEN}\nAssistant:"
                    f" {assistant_text}{END_OF_UTTERANCE_TOKEN}\n"
                )
            else:
                text += (
                    f"User: {random_linebreak}{user_text}{END_OF_UTTERANCE_TOKEN}\nAssistant:"
                    f" {assistant_text}{END_OF_UTTERANCE_TOKEN}\n"
                )
        # Remove trailing and leading whitespaces, including newlines and tabs
        text = text.strip("\n")

        sample_input_ids = tokenizer.encode(text, add_special_tokens=False)
        if add_end_of_doc_token:
            sample_input_ids += [tokenizer.eos_token_id]

        if add_begin_of_doc_token:
            sample_input_ids = [tokenizer.bos_token_id] + sample_input_ids

        if len(sample_input_ids) > max_seq_len:
            continue
        filtered_image_batch.append([image_transform(im) for im in images])
        filtered_input_ids.append(sample_input_ids)

    input_ids_to_pack = filtered_input_ids
    images_to_pack = filtered_image_batch

    (
        output_input_ids,
        output_labels,
        output_images,
        output_attention_masks,
        output_pixel_attention_masks,
        output_num_images,
        output_num_text_tokens,
    ) = greedy_packing(
        input_ids_to_pack=input_ids_to_pack,
        images_to_pack=images_to_pack,
        max_seq_len=max_seq_len,
        max_num_images=MAX_NUM_IMAGES_AFTER_SPLIT,
        image_seq_len=image_seq_len,
        pad_token_id=pad_token_id,
        fake_token_around_image_id=fake_token_around_image_id,
        image_token_id=image_token_id,
        double_breaking_lines_token_ids=double_breaking_lines_token_ids,
        output_input_ids=[],
        output_images=[],
        output_attention_masks=[],
        output_pixel_attention_masks=[],
        output_num_images=[],
        output_num_text_tokens=[],
        truncate_images_within_same_example=False,
        mask_labels=True,
        end_of_utterance_token_id=end_of_utterance_token_id,
        bos_token_id=tokenizer.bos_token_id if add_begin_of_doc_token else None,
        eos_token_id=tokenizer.eos_token_id if add_end_of_doc_token else None,
        assistant_token_ids=assistant_token_ids,
    )
    result = prepare_result_return(
        output_input_ids=output_input_ids,
        output_labels=output_labels,
        output_images=output_images,
        output_attention_masks=output_attention_masks,
        output_pixel_attention_masks=output_pixel_attention_masks,
        output_num_images=output_num_images,
        output_num_text_tokens=output_num_text_tokens,
    )
    return result
