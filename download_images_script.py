from typing import *
from tqdm import tqdm
import os
from pathlib import Path
import asyncio

from PIL import Image
import pandas as pd
import torchvision
import torch

# to communicate with Telegram (to retrieve images)
from telethon import TelegramClient, sync
from telethon.sessions import StringSession
from telethon.tl.types import InputMessagesFilterPhotos
from telethon.errors.rpcerrorlist import UsernameInvalidError, UsernameNotOccupiedError

# to compute a perceptual hash on an image
import imagehash
import torchvision

import db_utilities as utils

to_tensor = torchvision.transforms.ToTensor()


channel_ids: List[int] = utils.get_channel_ids()

# extension of the images to download
image_extensions = {'.png', '.jpg', '.jpeg', '.gif'}


def get_media_ids(channels: List[int]) -> Dict[Tuple[str, str], List[int]]:
    '''
    Input: list of channel ids for which to retrieve the media ids
    Output: dict from (channel_id,channel_name) to list of message ids of the photos
    '''
    # dict from (channel_id,channel_name) to list of message ids of the photos
    media_ids: Dict[Tuple[str, str], List[int]] = dict()

    # for each telegram channel
    for id in tqdm(channel_ids):
        channel: Dict = utils.get_channel_by_id(id)
        channel_name: str = channel['username']
        media_dict = channel['generic_media']
        # for each media in the channel
        for media in media_dict:
            if media_dict[media]['extension'] in image_extensions and media_dict[media]['media_id'] is not None:
                # retrieve the list of media ids for the channel
                ids = media_ids.get((id, channel_name), [])
                # append the new media id to the list
                ids.append(int(media))
                # update the dict
                media_ids[(id, channel_name)] = ids
    return media_ids


def hamming_distance(hash1: str, hash2: str) -> int:
    '''
    Given two hash hash1 and hash2, it computes the hamming distance between them
    (that is, the number of binary bits in wich they differ).
    '''
    # convert the hashes into binary strings
    b1 = bin(int(str(hash1), 16))[2:]  # remove the prefix '0b'
    b2 = bin(int(str(hash2), 16))[2:]  # remove the prefix '0b'
    # take the minimum and maximum length strings
    min_b = min(b1, b2, key=lambda x: len(x))
    max_b = b1 if b1 != min_b else b2
    # add zeros on top of the shortest to make it reach the size of the longest
    while len(min_b) < len(max_b):
        min_b = '0' + min_b
    diff = 0
    # count the number of different bits
    for bit in range(len(max_b)):
        if max_b[bit] != min_b[bit]:
            diff += 1
    return diff


# TODO count number of same images and perceptually same images
# TODO count number of None images (per channel)
# mapping from perceptual hash of image to # of times it is found
hash2same: Dict[str, int] = dict()
# mapping from perceptual hash of image to # of times a perceptually similar is found
hash2similar: Dict[str, int] = dict()
# mapping from hash of image to its path
hash2image: Dict[str, str] = dict()
# mapping from (channel_id, message_id) to hash of the image
channel2hash: Dict[Tuple[str, int], str] = dict()
# mapping from channel_id to a tuple (n,tot) where n is # of images not dowloaded and tot is the total
channel2NoneImages: Dict[str, Tuple[int, int]] = dict()


async def download_images(media_ids: Dict[Tuple[str, str], List[int]], folder: str = "images/") -> pd.DataFrame:
    for channel_id, channel_name in tqdm(media_ids.keys()):
        images_id = media_ids[(channel_id, channel_name)]
        channel2NoneImages[channel_id] = (0, len(images_id))
        images = client.iter_messages(
            channel_name, filter=InputMessagesFilterPhotos, limit=len(images_id), ids=images_id)
        i = 0
        try:
            async for image in images:  # for every image to download
                if image and image.photo:  # if there is an image to download
                    path = await client.download_media(image.photo, file=folder)
                    if path:  # if the image was downloaded, it returned its path
                        pil_image = Image.open(path)
                        hash = imagehash.average_hash(pil_image)
                        # get the extension of the image
                        extension: str = os.path.splitext(path)[1]
                        # call the file channelName_mediaId
                        new_path = str(Path(folder).joinpath(
                            channel_name+"_"+str(images_id[i]))) + extension
                        os.rename(path, new_path)
                        path = new_path
                        i += 1
                        if len(hash2image) > 0:
                            most_similar = min(
                                hash2image.keys(), key=lambda x: hamming_distance(x, hash))
                            ham_dist = hamming_distance(most_similar, hash)
                            if ham_dist < 5:
                                if ham_dist == 0:  # they might be exactly the same image
                                    sim_image = Image.open(
                                        hash2image[most_similar])
                                    # take the pixels of the images
                                    t_im: torch.Tensor = to_tensor(pil_image)
                                    t_sim: torch.Tensor = to_tensor(sim_image)
                                    sim_image.close()
                                    # they have exactly the same pixels
                                    if t_im.shape == t_sim.shape and (t_im != t_sim).sum().item() == 0:
                                        hash2same[most_similar] = hash2same.get(
                                            most_similar, 0) + 1
                                    else:  # they differ in at least one pixel
                                        # print(f"\n{path} and {hash2image[most_similar]} have same perceptual hash but different pixels\n")
                                        hash2similar[most_similar] = hash2similar.get(
                                            most_similar, 0) + 1
                                else:  # they are slightly different
                                    # print(f"\n{path} and {hash2image[most_similar]} have an hamming distance of {ham_dist}\n")
                                    hash2similar[most_similar] = hash2similar.get(
                                        most_similar, 0) + 1
                                os.remove(path)
                                hash = most_similar
                            else:
                                hash2image[str(hash)] = path
                        else:
                            hash2image[str(hash)] = path
                        pil_image.close()
                        channel2hash[(channel_id, image.id)] = str(hash)
                else:
                    n, tot = channel2NoneImages[channel_id]
                    channel2NoneImages[channel_id] = (n+1, tot)
        # (UsernameInvalidError, UsernameNotOccupiedError) as e:
        except Exception as e:
            print(e)
            continue


def build_dfs() -> Tuple[pd.DataFrame, pd.DataFrame]:
    # convert hash2NoneImages to a dataframe
    df_channels = pd.DataFrame.from_dict(
        channel2NoneImages, orient='index', columns=['n', 'tot'])
    df_channels['channel'] = df_channels.index
    # now remove the index
    df_channels.reset_index(drop=True, inplace=True)
    df_channels['ratio'] = df_channels['n']/df_channels['tot']

    # convert hash2same, hash2similar and hash2image to dataframe
    df_same = pd.DataFrame.from_dict(
        hash2same, orient='index', columns=['#found'])
    df_same['hash'] = df_same.index
    df_same.reset_index(drop=True, inplace=True)
    df_similar = pd.DataFrame.from_dict(
        hash2similar, orient='index', columns=['#similar'])
    df_similar['hash'] = df_similar.index
    df_similar.reset_index(drop=True, inplace=True)
    df_image = pd.DataFrame.from_dict(
        hash2image, orient='index', columns=['path'])
    df_image['hash'] = df_image.index
    df_image.reset_index(drop=True, inplace=True)
    # join the 3 dataframes into one on the hash column
    df_images = df_same.merge(df_similar, on='hash', how='outer').merge(
        df_image, on='hash', how='outer')
    df_images['#similar'] = df_images['#similar'].fillna(0)
    df_images['#found'] = df_images['#found'].fillna(0)

    return df_channels, df_images


media_ids = get_media_ids(channel_ids)
# TOKEN = "6024785388:AAEUnoCjCS_rE6inRrpUp1Xw_HqQ2I9Aoa8"
api_id = "21418053"
api_hash = "8816a0370ace42cf311ebb554450c30c"
client = TelegramClient('session_name', api_id, api_hash).start()

loop = asyncio.get_event_loop()
loop.run_until_complete(download_images(media_ids))

df_channels, df_images = build_dfs()
# save the dataframes to csv
df_channels.to_csv("channels.csv", index=False)
df_images.to_csv("images.csv", index=False)
