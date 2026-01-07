from SoccerNet.Downloader import SoccerNetDownloader

downloader = SoccerNetDownloader(LocalDirectory="/workspace/datasets/")
downloader.password = "s0cc3rn3t"
downloader.downloadDataTask(task="spotting-OSL", split=["train", "valid", "test", "challenge"], version="ResNET_PCA512")
