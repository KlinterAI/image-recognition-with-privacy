#!/usr/bin/sh

python3 ./text_spotting_demo.py \
-m_m "../models/intel/text-spotting-0005/text-spotting-0005-detector/FP16/text-spotting-0005-detector.xml" \
-m_te "../models/intel/text-spotting-0005/text-spotting-0005-recognizer-encoder/FP16/text-spotting-0005-recognizer-encoder.xml" \
-m_td "../models/intel/text-spotting-0005/text-spotting-0005-recognizer-decoder/FP16/text-spotting-0005-recognizer-decoder.xml" \
-i "./inference_images/text_inference.png"

