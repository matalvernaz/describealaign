# describealaign

> **Note:** describealaign is a maintained fork of [describealign](https://github.com/julbean/describealign) by Julian Brown. It carries fixes for PAL/NTSC source mismatches, commercial-break seam crossfades, different-cut sources (pass-2 anchoring to confirmed matches, so an unrated video against a theatrical description no longer plays narration early), a bounded mux interleave window (remuxes with many sparse subtitle tracks no longer make ffmpeg buffer the whole file), structured (`--json`) output, parsed-audio caching, and a property-based test suite. License remains GPLv3.

Combines videos with matching audio files (e.g. audio descriptions). Works by aligning parts of the audio file to matching parts of the video's sound.


## Quickstart

Create a copy of a video file with the sound replaced by an audio description:

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/describealaign_gui_main.PNG" alt="GUI main" align="middle" width="50%"/>

Select a video file and a corresponding audio description using the file browsers, then click "Combine":

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/describealaign_gui_combiner.PNG" alt="GUI combiner" align="middle" width="50%"/>

The combined media is saved in the folder "videos_with_ad" placed in the home directory. The output directory and other settings can be changed in "Settings":

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/describealaign_gui_settings.PNG" alt="GUI settings" align="middle" width="50%"/>


## Installation

### binary method

Windows, Mac, and Linux users can download and unzip the [latest release](https://github.com/matalvernaz/describealaign/releases/latest), then double click on describealaign.exe (Windows) or the describealaign executable to open the GUI.

Note for Mac binary users: To open the binary, you'll need to ctrl+click (or right click) on the binary, then click "Open" and then click "Open" again in the window that pops up.

### script method

The python script (describealaign.py) can be downloaded and run directly (with Python 3.8+) after installing the dependencies:
```bash
pip install -r requirements.txt
python3 describealaign.py
```


## Testing Installation

The installation can be tested on a clip from the 1929 comedy short [Ask Dad](https://archive.org/details/ask_dad), with the first part of an [audio description](https://archive.org/details/MoviesForTheBlind01-askDad) provided by Valerie H. in her podcast [Movies For the Blind.](https://moviesfortheblind.com/) Download the trimmed versions from the test_media folder in this repository, then select them in the GUI:

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/describealaign_gui_main_filled.PNG" alt="GUI main filled" align="middle" width="50%"/>

This produces two outputs, a new video file "videos_with_ad/ad_ask_dad_trimmed.mp4" and a plot in alignment_plots:

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/ask_dad_trimmed.png" alt="Ask Dad Trimmed Alignment" align="middle" width="50%"/>

The plot shows the audio description starts 202 seconds before the video, which means Valerie starts describing Ask Dad 202 seconds into the podcast. After 40 more seconds, the podcast skips ahead by 3 seconds.

If the full video (22 minutes) and audio description (27 minutes) are used instead, describealaign runs in about 30 seconds, using up about 630 MB of RAM, and we get the following plot:

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/ask_dad.png" alt="Ask Dad Alignment" align="middle" width="50%"/>

This plot shows a number of small skips starting around 10 minutes in, which add up to a total offset of 30 seconds by the end of the video.

A text version of each plot is saved alongside it:

```
Parameters: {'stretch_audio': False, 'no_pitch_correction': False}
Version Hash: 8079dedc
Input file similarity: 50.70%
Main changes needed to video to align it to audio input:
Start Offset: 201.81 seconds
Median Rate Change: 0.00%
Rate change of     -0.0% from  0:00:00.000 to  0:00:37.328 aligning with audio from  0:03:21.810 to  0:03:59.138
Rate change of     21.9% from  0:00:37.328 to  0:00:52.772 aligning with audio from  0:03:59.138 to  0:04:11.810
Rate change of      0.0% from  0:00:52.772 to  0:02:58.581 aligning with audio from  0:04:11.810 to  0:06:17.619
```


## Advanced Usage

### batching

describealaign can be given a directory of videos and a directory of audio files rather than individual files. describealaign assumes files from the two directories correspond based on their lexicographic order.

Alternatively, multiple files can be selected simultaneously in the GUI's file selectors. The selected Video and Audio files are first sorted in lexicographic order, then corresponded and aligned.

### drag-and-drop

describealaign supports dragging and dropping files or folders onto the video and audio input lists. Files with irrelevant extensions are ignored, while valid files appear in lexicographic order.

### dark mode

describealaign opens in either light or dark mode depending on the OS's desktop theme.

### stretch_audio (audio-to-video alignment)

By default describealaign stretches video to fit audio descriptions, but the inverse is also possible: stretching the audio description to fit the video with the "--stretch_audio" argument. In both modes, all original audio tracks from the video are preserved in the output alongside the audio description track (which is set as the default).

The alignment listens to the video's first audio track, and the description's gaps are filled from it. When that track is a dub (some "Dual Audio" releases put a foreign-language track first), pass "--audio_stream N" to use another, counting audio tracks from 0: "--audio_stream 1" for the second. Without "--ad_language", the description track then takes that track's language.

When using --stretch_audio, the plot also shows which segments of audio were replaced:

<img src="https://github.com/matalvernaz/describealaign/blob/main/readme_media/ask_dad_stretch_audio.png" alt="Ask Dad Stretch Audio Alignment" align="middle" width="50%"/>

The original audio is used for segments that would be too noticeably distorted (i.e. more than 10% stretched).

### dry run

Running with `--dry-run` performs the full alignment and saves the alignment plot, but skips writing the output media file. Useful for quickly checking alignment quality before committing to a slow stretch operation:
```bash
describealaign video.mp4 audio_desc.mp3 --dry-run
```

### watch mode

Running with `--watch` monitors the input directories and automatically processes new file pairs as they appear. Useful for automated pipelines (e.g. processing audio descriptions as they are downloaded):
```bash
describealaign /videos /audio_descriptions --watch
describealaign /videos /audio_descriptions --watch --watch-interval 60
```

The `--watch-interval` option controls how often the directories are scanned in seconds (default: 30). Already-processed files are skipped on each scan. Press Ctrl+C to stop.

### audio-to-audio

Whereas describealaign is designed to align video-to-audio, it can also align an audio file to another audio file.

### command line interface

describealaign can be run without the GUI by specifying input media as positional arguments:
```bash
describealaign video.mp4 audio_desc.mp3
```

Note for Mac binary users: The executable is inside the .app and can be run from Terminal with:
```
describealaign.app/Contents/MacOS/describealaign video.mp4 audio_desc.mp3
```

### module

describealaign can also be used as a python module:
```python
import describealaign as da
da.combine("ask_dad_trimmed.mp4", "ask_dad_moviesfortheblind_ep_01_trimmed.mp3")
```


## Interesting Use Cases

### dub alignment

describealaign is robust enough to align media with completely different dialogue, meaning it can align audio dubbed in a different language to the original video.

### lossless video editing

With default settings (i.e. --stretch_audio set to False), describealaign doesn't re-encode either the video or audio streams. It aligns them by modifying the timestamps that video frames are shown at, which means no loss in quality. Basic video editing can be done by deleting or stretching segments of a video's sound in Audacity, then running describealaign on the original video and the modified audio.
