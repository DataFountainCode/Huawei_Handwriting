#!/usr/bin/env bash
python resnet_crnn_main.py --trainroot ./to_lmdb/train_index_all --valroot ./to_lmdb/test_index --GPU_ID ${1}
