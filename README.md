# About Pull Request

This repository is read-only, so Pull Request is not accepted. Thank you for your understanding.

# Caution

If you use this software for commercial, you must pay fees to NVIDIA.

# Antares

Antares is a shogi engine, written by C#.

Shogi is game like chess.

## Source Code Explanation

This software is large scale, so I can not explain about details. So, I summarize below.

This software can analyze shogi records in four different methods.

(1) Monte Carlo search with deep convolutional neural network.

(2) Normal alpha-beta search.

(3) Alpha-beta search with realization probability.

(4) Monte Carlo search with shallow layers.

## Deep Neural Network

Deep counvolutional neural networks used in (1) above, are learned with using PyTorch. For avoiding bugs, input features are very simple. This software uses ONNX.

## Alpha-Beta Search

Alpha-Beta Search is adopted at (2) and (3). The structure of (2) and (3) are not much influenced by StockFish. (2) is almost my unique implementation. (3) is 
influenced by Gekisashi very much.

## Feature Vectors in Alpha-Beta Search

King-King-Piece and King-Piece-Piece are adopted int (2) and (3). This features are derived from Bonanza, Apery and YaneuraOu. It's a subtle difference, The feature vectors in this software does not include turn but includes promoted golds.

## Prediction Accuracy
(1)'s prediction accuracy is good but not marvelous. (2) and (3) are not good.
(4) is bad. The fact explains the difference of experts and I.

## Operating environment

(1) OS: Windows 11 Pro

(2) .NET Version: 9.0

(3) Microsoft.ML.OnnxRuntime.Gpu Version: 1.24.4

(4) TorchSharp-cuda-windows Version: 0.105.0

## How to build

Double click "Antares.sln" and build with using Visual Studio. I identified this software is running in debug build and do not identified in release build.

go build

## References

I developed this software referring to the softwares as below.

(1) Bonanza

(2) Apery

(3) YaneuraOu

(4) Gikou

(5) dlshogi

As far as I know, the source code for Bonanza and dlshogi is currently not publicly available.

Then, I developed this software referring to the books as below.

(1) 山岡忠夫(2018),『将棋AIで学ぶディープラーニング』マイナビ出版 

(2) 山岡忠夫、加納邦彦(2021), 『強い将棋ソフトの創りかた　Pythonで実装するディープラーニング将棋AI』マイナビ出版

(3) 大槻知史(著)、三宅陽一郎(監修)(2018), 『最強囲碁AI アルファ碁解体新書　増補改訂版』翔泳社

(4) 原田達也(2017), 機械学習プロフェッショナルシリーズ『画像認識』講談社

(5) コンピュータ将棋協会監修、滝澤武信、松原仁、小谷善行他著(2012), 『人間に勝つコンピュータ将棋の作り方　あから2010を生み出したアイデアと工夫の軌跡』技術評論社
(6) 松原仁　編著(2005), 『コンピュータ将棋の進歩5』共立出版

(7) 松原仁　編著(2012), 『コンピュータ将棋の進歩6』共立出版

(8) 海野裕也、岡野原大輔、得居誠也、徳永拓之(2015), 機械学習プロフェッショナルシリーズ『オンライン機械学習』

(9) 鈴木大慈(2015), 機械学習プロフェッショナルシリーズ『確率的最適化』

(10) 金森敬文、鈴木大慈、竹内一郎、佐藤一誠(2016), 機械学習プロフェッショナルシリーズ『機械学習のための連続最適化』

(11) 金森敬文(2015), 機械学習プロフェッショナルシリーズ『統計的学習理論』

(12) 岡谷貴之(2022), 機械学習プロフェッショナルシリーズ『深層学習』 改訂第2版

I referred to the URL below.

https://www.chessprogramming.org/Main_Page
