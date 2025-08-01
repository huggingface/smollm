import torch
from torch import nn
from transformers import Idefics3Model, Idefics3ForConditionalGeneration
from typing import Dict, Any, List, Optional, Union, Tuple
from transformers.cache_utils import Cache, DynamicCache

from transformers.utils import add_start_docstrings_to_model_forward, logging
from transformers.models.idefics3.modeling_idefics3 import IDEFICS3_INPUTS_DOCSTRING, Idefics3BaseModelOutputWithPast

logger = logging.get_logger(__name__)

class SmolLMModel(Idefics3Model):
    """
    A subclass of Idefics3Model. We do *not* remove or block the call to inputs_merger
    in forward. Instead, we override inputs_merger here with custom logic.
    """
    def __init__(self, config):
        super().__init__(config)
        self.frames_per_clip = getattr(config, "frames_per_clip", 1)

    def inputs_merger(
        self,
        input_ids: torch.LongTensor,
        inputs_embeds: torch.Tensor,
        image_hidden_states: torch.Tensor
    ) -> torch.Tensor:
        """
        Merge text embeddings with image embeddings out-of-place (no in-place indexing).
    
        The shapes are something like:
          - input_ids:          (B, T)
          - inputs_embeds:      (B, T, D)
          - image_hidden_states:(N, S, D) where N is total images across the batch,
            S is #patches (or #slots) per image, D is embedding dim.
    
        Logic:
          1) For each sample in the batch, find <image> tokens in the text.
          2) If zero <image> tokens => text-only. Concatenate a zero-length slice
             from image_hidden_states but do NOT advance the offset. This ensures
             the model's image encoder is still in the computation graph, but we
             skip "consuming" any image block for a text-only sample.
          3) If there are <image> tokens, they appear in multiples of S for each image
             (because each image is S embeddings). We chunk those positions into groups
             of S. For each chunk => we consume one block from image_hidden_states[offset]
             (which is shape (S, D)), and place each row into the text in place of a token.
    
        Returns:
          A tensor of (B, T, D).
        """
    
        ##############################################
        # 1) Basic shape checks
        ##############################################
        #old_merger_outputs = self.inputs_merger_old(input_ids, inputs_embeds, image_hidden_states)
        B, T, D_text = inputs_embeds.shape
        N, S, D_img  = image_hidden_states.shape
        if D_text != D_img:
            raise ValueError(
                f"Text embedding dim {D_text} != image embedding dim {D_img}"
            )
    
        ##############################################
        # 2) We'll track how many images we've used so far across the entire batch
        ##############################################
        image_offset = 0
    
        # We'll store one merged tensor per batch sample
        merged_outputs: List[torch.Tensor] = []
    
        ##############################################
        # 3) Iterate through each sample
        ##############################################
        for b_idx, (cur_ids, cur_embeds) in enumerate(zip(input_ids, inputs_embeds)):
            # Find positions of <image> tokens in the text
            image_positions = (cur_ids == self.image_token_id).nonzero(as_tuple=True)[0]
            num_image_tokens = len(image_positions)
    
            # If no <image> => text-only
            if num_image_tokens == 0:
                # We do not consume any row from image_hidden_states; 
                # but we do a zero-length slice so the image encoder is in the graph.
                empty_slice = image_hidden_states[0][:0, :]  # shape (0, D)
                # Concatenate text plus that empty slice.
                # NOTE: this is important for DeepSpeed.
                merged_text_only = torch.cat([cur_embeds, empty_slice], dim=0)
                merged_outputs.append(merged_text_only)
                continue
    
            # Otherwise, we have at least one <image> token.
            # Typically, if each image is S embeddings, we expect the total # of <image> tokens
            # in this sample to be multiple of S => each group of S tokens = 1 image
            if num_image_tokens % S != 0:
                raise ValueError(
                    f"Sample {b_idx} has {num_image_tokens} <image> tokens, not a multiple of S={S}. "
                    "Cannot map them to blocks of shape (S, D)."
                )
    
            # We'll chunk image_positions into groups of size S
            positions_list = image_positions.tolist()
            # Example: if num_image_tokens=162 and S=81 => we have 2 images => 2 chunks each of length 81
            chunks = [
                positions_list[i : i + S]
                for i in range(0, num_image_tokens, S)
            ]
    
            # We'll build a list of segments: text, then image row(s), text, etc.
            segments = []
            text_start = 0
    
            # For each chunk (each chunk => 1 image)
            for chunk in chunks:
                # image_hidden_states[image_offset] => shape (S, D)
                cur_block = image_hidden_states[image_offset]
                image_offset += 1
    
                # We'll iterate over the S positions in ascending order
                for i_s, pos in enumerate(chunk):
                    # Add text from [text_start..pos)
                    if pos > text_start:
                        segments.append(cur_embeds[text_start:pos])
                    # Then add one row from cur_block => shape (1, D)
                    row_of_block = cur_block[i_s : i_s + 1, :]
                    segments.append(row_of_block)
                    # skip the <image> token
                    text_start = pos + 1
    
            # leftover text after the final <image> token
            if text_start < T:
                segments.append(cur_embeds[text_start:])
    
            # cat them into a single (T_b, D) tensor
            merged_sample = torch.cat(segments, dim=0)
            merged_outputs.append(merged_sample)
            
        merged_outputs = torch.stack(merged_outputs)
        #assert (old_merger_outputs==merged_outputs).all()
        return merged_outputs


    @add_start_docstrings_to_model_forward(
        """
        Inputs fed to the model can have an arbitrary number of images. To account for this, pixel_values fed to
        the model have image padding -> (batch_size, max_num_images, 3, max_heights, max_widths) where
        max_num_images is the maximum number of images among the batch_size samples in the batch.
        Padding images are not needed beyond padding the pixel_values at the entrance of the model.
        For efficiency, we only pass through the vision_model's forward the real images by
        discarding the padding images i.e. pixel_values of size (image_batch_size, 3, height, width) where
        image_batch_size would be 7 when num_images_per_sample=[1, 3, 1, 2] and max_num_images would be 3.
        """,
        IDEFICS3_INPUTS_DOCSTRING,
    )
    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        pixel_attention_mask: Optional[torch.BoolTensor] = None,
        image_hidden_states: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, Idefics3BaseModelOutputWithPast]:
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.training and self.text_model.gradient_checkpointing and use_cache:
            logger.warning_once(
                "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`..."
            )
            use_cache = False

        # retrieve input_ids and inputs_embeds
        if input_ids is not None:
            batch_size, seq_length = input_ids.shape
        elif inputs_embeds is not None:
            batch_size, seq_length, _ = inputs_embeds.shape
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        past_seen_tokens = 0
        if use_cache:
            if past_key_values is None:
                past_key_values = DynamicCache()
            past_seen_tokens = past_key_values.get_seq_length()

        if inputs_embeds is not None and input_ids is None and past_seen_tokens == 0:
            raise ValueError("When first calling the model, if input_embeds are passed, input_ids should not be None.")

        if inputs_embeds is None:
            inputs_embeds = self.text_model.get_input_embeddings()(input_ids).to(self.device)

        # START VISUAL INPUTS INTEGRATION
        if pixel_values is not None and image_hidden_states is not None:
            raise ValueError("You cannot specify both pixel_values and image_hidden_states at the same time")
        elif pixel_values is not None:
            batch_size, num_images, num_channels, height, width = pixel_values.shape
            pixel_values = pixel_values.to(dtype=self.dtype)  # fp16 compatibility
            pixel_values = pixel_values.view(batch_size * num_images, *pixel_values.shape[2:])

            # Remove padding images - padding images are full 0.
            nb_values_per_image = pixel_values.shape[1:].numel()
            real_images_inds = (pixel_values == 0.0).sum(dim=(-1, -2, -3)) != nb_values_per_image
            
            if not any(real_images_inds):
                # no images, leave one empty image.
                real_images_inds[0] = True
                
            pixel_values = pixel_values[real_images_inds].contiguous()
            
            # Handle the vision attention mask
            if pixel_attention_mask is None:
                pixel_attention_mask = torch.ones(
                    size=(pixel_values.size(0), pixel_values.size(2), pixel_values.size(3)),
                    dtype=torch.bool,
                    device=pixel_values.device,
                )
            else:
                # Remove padding images from the mask
                pixel_attention_mask = pixel_attention_mask.view(
                    batch_size * num_images, *pixel_attention_mask.shape[2:]
                )
                pixel_attention_mask = pixel_attention_mask[real_images_inds].contiguous()

            patch_size = self.config.vision_config.patch_size
            patches_subgrid = pixel_attention_mask.unfold(dimension=1, size=patch_size, step=patch_size)
            patches_subgrid = patches_subgrid.unfold(dimension=2, size=patch_size, step=patch_size)
            patch_attention_mask = (patches_subgrid.sum(dim=(-1, -2)) > 0).bool()

            # Get sequence from the vision encoder
            image_hidden_states = self.vision_model(
                pixel_values=pixel_values,
                patch_attention_mask=patch_attention_mask,
            ).last_hidden_state
            
            # Modality projection & resampling
            image_hidden_states = self.connector(image_hidden_states)

            N, S, D = image_hidden_states.shape
            # We'll rebuild a list for each sample
            new_image_embeds = []
            offset = 0
            real_inds_list = real_images_inds.nonzero(as_tuple=True)[0]
            # total real images across batch
            total_real = real_inds_list.size(0)

            for b_idx in range(batch_size):
                # count how many <image> tokens are in text
                if input_ids is not None:
                    cur_ids = input_ids[b_idx]
                    n_image_tokens = (cur_ids == self.image_token_id).sum().item()
                else:
                    n_image_tokens = 0
                    
                start_im = b_idx * num_images
                end_im   = start_im + num_images
                valid_count = real_images_inds[start_im:end_im].sum().item()

                # now offset.. we only apply those valid_count from the global offset
                cur_ihs = image_hidden_states[offset : offset + valid_count]
                offset += valid_count

                # If mismatch => should be video, do frames average
                # frames_per_clip => how many images belong to 1 "logical" image
                # e.g. if valid_count = 8, frames_per_clip=4 => 2 images
                # or if n_image_tokens != valid_count => treat it as needing frames avg
                if (n_image_tokens != valid_count * self.image_seq_len) and (valid_count > 1) and (self.frames_per_clip > 1):
                    if valid_count % self.frames_per_clip != 0:
                        raise ValueError(
                            f"Batch {b_idx}: mismatch in frames/clip AND {valid_count} not divisible by {self.frames_per_clip}"
                        )
                    logical_imgs = valid_count // self.frames_per_clip
                    cur_ihs = cur_ihs.view(logical_imgs, self.frames_per_clip, S, D)
                    cur_ihs = cur_ihs.mean(dim=1)  # => shape: (logical_imgs, S, D)
                    # Log if you want
                    # print(f"[Video logic] Sample {b_idx}: #image_tokens={n_image_tokens} vs #images={valid_count}, => chunk & avg.")
                new_image_embeds.append(cur_ihs)

            # Now we cat them back
            if len(new_image_embeds) > 0:
                image_hidden_states = torch.cat(new_image_embeds, dim=0) if len(new_image_embeds) > 1 else new_image_embeds[0]
            else:
                # no images at all
                image_hidden_states = image_hidden_states[:0]

        elif image_hidden_states is not None:
            image_hidden_states = image_hidden_states.to(dtype=self.dtype, device=input_ids.device)

        if past_seen_tokens == 0 and inputs_embeds is not None and image_hidden_states is not None:
            # When we generate, we don't want to replace the potential image_token_id that we generated by images
            # that simply don't exist

            inputs_embeds = self.inputs_merger(
                input_ids=input_ids,
                inputs_embeds=inputs_embeds,
                image_hidden_states=image_hidden_states,
            )

        outputs = self.text_model(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        if not return_dict:
            return tuple(v for v in [*outputs, image_hidden_states] if v is not None)

        return Idefics3BaseModelOutputWithPast(
            last_hidden_state=outputs.last_hidden_state,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            image_hidden_states=image_hidden_states,
        )




class SmolLMMForConditionalGeneration(Idefics3ForConditionalGeneration):
    """
    A subclass of Idefics3ForConditionalGeneration that uses MyIdefics3Model
    instead of the default Idefics3Model.
    """

    def __init__(self, config):
        super().__init__(config)
        # Instead of the original self.model = Idefics3Model(config),
        # we point to our custom class.
        self.model = SmolLMModel(config)

        # We *keep* the same lm_head from the parent, or re-init if you prefer:
        self.lm_head = nn.Linear(
            config.text_config.hidden_size, config.text_config.vocab_size, bias=False
        )

        # If parent sets up any post_init() logic:
        self.post_init()