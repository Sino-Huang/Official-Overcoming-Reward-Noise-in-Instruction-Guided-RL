import torch as th 
from transformers import AutoModel, AutoTokenizer

# ! the inputs has 3 keys, 'input_ids', 'attention_mask', 'pixel_values']
# 'input_ids', 'attention_mask' can be obtained from tokenizer
# for clip, the shape of 'pixel_values' is (batch_size, 3, 224, 224)
# image_embeds.shape  [batch, 512] and text_embeds.shape is [num_texts, 512]
# for xclip the shape of 'pixel_values' is (batch_size, 8, 3, 224, 224) where the 8 is the frame length,
# ! the shapes are bit different, in xclip, the outputs['video_embeds'].shape is [batch_size, 512],
# ! the outputs['text_embeds'].shape is [batch_size, num_texts, 512]
# if batch_size == num_texts, you can make the data shape be [batch_size, 512] and [batch_size, 512], the cosine similarity shape will be torch.Size([2]),
# if you want to measure the similarity between each text and each image, make sure you unsqueeze the embeddings 


def get_pretrained_vision_language_model(pretrained_model_cls):
    
    model = AutoModel.from_pretrained(pretrained_model_cls)
    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_cls)
    
    return model, tokenizer


def postprocess_clip_outputs(outputs):
    vision_embeds = outputs['image_embeds'] # shape [batch, 512]
    text_embeds = outputs['text_embeds'] # shape [num_texts, 512]
    return vision_embeds, text_embeds # ensure the output shape is [batch, 512] and [num_texts, 512]


def postprocess_xclip_outputs(outputs):
    vision_embeds = outputs['video_embeds'] # shape [batch, 512]
    text_embeds = outputs['text_embeds'] # shape [batch, num_texts, 512]
    text_embeds = text_embeds[0] # shape [num_texts, 512]
    return vision_embeds, text_embeds 
    
    

