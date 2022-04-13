#!/usr/bin/env python3

import sys
from PySide6.QtCore import QStandardPaths, Qt, Slot, QPoint, QRect, QSize, QLine
from PySide6.QtGui import QAction, QIcon, QKeySequence, QScreen, QPixmap, QPainter, QPen, QColor, QBrush, QPolygon, QFont
from PySide6.QtWidgets import (QApplication, QDialog, QFileDialog,
                               QMainWindow, QSlider, QStyle, QToolBar, QHBoxLayout, QVBoxLayout, QGridLayout, QWidget, QLabel, QComboBox, QGroupBox, QCheckBox, QLayout, QScrollArea, QRadioButton, QPushButton, QStatusBar, QSpinBox)
from PySide6.QtMultimedia import (QAudio, QAudioOutput, QMediaFormat,
                                  QMediaPlayer)
from PySide6.QtMultimediaWidgets import QVideoWidget
from PySide6.QtWebEngineWidgets import QWebEngineView
from PySide6.QtWebEngineCore import QWebEngineProfile
from PySide6.QtNetwork import QNetworkCookie

import glob
import os
from pathlib import Path
import json
import re
import functools
from shapely.geometry import Polygon
import copy
import math

CS_DIR = "/mrtstorage/datasets/public/cityscapes"
VID_DIR = "/data/cityscapes_videos"
TL_DIR = "/home/janosovits/cityscapes_labelling/labels_tls/extended-cityscapes-labels/gtFine"
DEPTH_FILE = "/home/janosovits/cityscapes_labelling/labels_tls/all_boxes.txt"
DEPTH_PREFIX = "/home/janosovits/cityscapes_labelling/labels_tls/extended-cityscapes-labels"

SIZE_FACTOR_IMG = 1.0
IMG_SZ = (2048, 1024)

VIDEO_ENDING = "_leftImg8bit.mp4"
IMG_ENDING = "_leftImg8bit.png"
LABEL_ENDING = "_gtFine_polygons.json"
VEHICLE_ENDING = "_vehicle.json"
TL_CLS = "8"

STATE_DICT = {"R": "red", "RY": "red-yellow", "Y" : "yellow", "G": "green", "O": "off", "U": "unknown"}
COLOR_DICT = {"R": "salmon", "RY": "orange", "Y": "yellow", "G": "lightgreen", "O": "grey", "U": "lightgrey"}
TYPE_DICT = {"Car": "car", "Ped": "pedestrian", "Bike": "bicycle", "Train": "train", "Bus": "bus", "CW": "car_warning", "Unk": "unknown"}
Q_COLOR_DICT = {"red": Qt.red, "red-yellow": Qt.magenta, "yellow": Qt.yellow, "green": Qt.green, "off": Qt.black, "unknown": Qt.gray}

def split_fn(fn):
    split, city, img = fn.split("/")
    return split, city, Path(img).stem.replace("_leftImg8bit", "")


def extract_lights(json_file):
    res = {}
    with open(json_file, "r") as f:
        root = json.load(f)
        for idx, obj in enumerate(root["objects"]):
            if obj["label"] == "traffic light" and not ("deleted" in obj.keys() and int(obj["deleted"]) != 0):
                res[str(idx)] = obj
    return res

def to_qpolygon(obj):
    qpoints = [QPoint(p[0], p[1]) for p in obj]
    return QPolygon.fromList(qpoints)

def ensure_dir(d):
    Path(d).parent.mkdir(parents=True, exist_ok=True)

def buffer(poly, buffer):
    p = Polygon(poly)
    dilated = p.buffer(buffer)
    x,y = dilated.exterior.coords.xy
    res = zip(x,y)
    return res

def to_bbox(poly):
    x, y = zip(*poly)
    return (min(x), min(y)), (max(x), max(y))

def pad_box(box, padding, img_sz):
    mmin ,mmax = box
    return (max(0, mmin[0] - padding), max(0, mmin[1] - padding)), (min(img_sz[0], mmax[0] + padding), min(img_sz[1], mmax[1] + padding))

def get_color(attrs, alpha=180):
    type_lookup = {"car": Qt.cyan, "pedestrian": Qt.green, "bicycle": Qt.yellow, "train": Qt.white, "bus": Qt.white, "unknown": Qt.gray, "car_warning": Qt.cyan}
    color = type_lookup[attrs["type"]]
    if attrs["relevant"] == "yes":
        color = Qt.magenta
    color = QColor(color)
    if attrs["visible"] == "no":
        color = color.darker()
    color.setAlpha(alpha)
    return color

def to_truth_vec(names, keys):
    return [True if n in keys else False for n in names]

def get_centroid(poly):
    return Polygon(poly).centroid.coords[:][0]

class DepthHolder():
    def __init__(self, depth_file):
        self._depths = {}
        cur_img = ""
        to_skip = len(DEPTH_PREFIX + "/gtFine/")
        with open(depth_file, "r") as f:
            for line in f.readlines():
                fn, cls, x1, x2, y1, y2, depth, idx = line.split()

                fn = fn[to_skip:]
                #print(fn)
                key = fn[:-len(LABEL_ENDING)]
                idx = int(idx)
                if key != cur_img:
                    self._depths[key] = {}
                    cur_img = key
                if cls != TL_CLS:
                    continue
                self._depths[key][idx] = float(depth) if math.isfinite(float(depth)) else 0
        print(self._depths.keys())

    def get_depth(self, city, idx):
        return self._depths[city][idx]

    def get_key(self, key):
        return self._depths[key]

class LabelIO():
    def __init__(self, file, depth_data):
        self._file = file
        self._state = None
        self._depth_data = depth_data
        if not os.path.exists(file):
            raise RuntimeError("Could not find " + file)
        with open(file, "r") as f:
            self._state = json.load(f)
        self._validate()

    def _set(self, idx, name, value):
        if self._state["objects"][idx]["label"] != "traffic light":
            raise ValueError("Idx {} not a traffic light".format(idx))
        self._state["objects"][idx]["attributes"][name] = value

    def _get(self, idx, name):
        return self._state["objects"][idx]["attributes"][name]

    def set_type(self, idx, t):
        self._set(idx, "type", t)

    def set_state(self, idx, t):
        self._set(idx, "state", t)

    def set_relevant(self, idx):
        new_val = "yes" if self._get(idx, "relevant") == "no" else "no"
        self._set(idx, "relevant", new_val)

    def set_lane_relevant(self, idx):
        new_val = "yes" if self._get(idx, "relevant") == "no" else "no"
        self._set(idx, "lane_relevant", new_val)

    def set_visible(self, idx):
        new_val = "yes" if self._get(idx, "visible") == "no" else "no"
        self._set(idx, "visible", new_val)

    def set_depth(self, idx, depth):
        self._set(idx, "depth", depth)

    def write(self):
        with open(self._file, "w") as f:
            json.dump(self._state, f, indent=4, sort_keys=True, allow_nan=False)

    def get_lights(self):
        res = {}
        for idx, obj in enumerate(self._state["objects"]):
            if obj["label"] == "traffic light" and not ("deleted" in obj.keys() and int(obj["deleted"]) != 0):
                res[idx] = obj
                res[idx]["depth_metric"] = self._depth_data[idx] if idx in self._depth_data.keys() else 0
        return res

    def delete_by_idx(self, tl_idx):
        del self._state["objects"][tl_idx]

    def _validate(self):
        for idx, l in self.get_lights().items():
            if not "visible" in l["attributes"]:
                l["attributes"]["visible"] = "no"
            if not "relevant" in l["attributes"]:
                l["attributes"]["relevant"] = "no"
            if not "state" in l["attributes"]:
                l["attributes"]["state"] = "unknown"
            if not "type" in l["attributes"]:
                l["attributes"]["type"] = "unknown"
            if not "lane_relevant" in l["attributes"]:
                if l["attributes"]["relevant"] == "yes":
                    l["attributes"]["lane_relevant"] = "yes"
                else:
                    l["attributes"]["lane_relevant"] = "unknown"
            if not "depth" in l["attributes"]:
                l["attributes"]["depth"] = 0


class DataLoader():
    def __init__(self, cs_dir=CS_DIR, vid_dir=VID_DIR, tl_dir=TL_DIR):
        self._cs_dir = cs_dir
        self._img_dir = os.path.join(cs_dir, "leftImg8bit")
        self._label_dir = os.path.join(cs_dir, "gtFine")
        self._vehicle_dir = os.path.join(cs_dir, "vehicle")
        self._vids_dir = vid_dir
        self._tl_dir = tl_dir
        self._available = {}
        self._discover()
        self._init_idx()

    def _discover(self):
        os.chdir(self._img_dir)
        imgs = glob.glob("*/*/*.png")
        for i in imgs:
            split_n, city_n, key = split_fn(i)
            city = "{}/{}".format(split_n, city_n)
            if not city in self._available.keys():
                self._available[city] = []
            self._available[city].append(key)
        for city in self._available.keys():
            self._available[city].sort()

    def _init_idx(self):
        self._cur_city = list(self._available.keys())[0]
        self._cur_idx = self._available[self._cur_city][0]

    def set_city(self, city):
        if not city in self._available.keys():
            raise ValueError("City " + city + " not found")
        self._cur_city = city
        self._cur_idx = self._available[self._cur_city][0]

    def set_idx(self, idx):
        if not idx in self._available[self._cur_city]:
            raise ValueError("Idx " + idx + " not found")
        self._cur_idx = idx

    def next_img(self):
        self._cur_idx = self.get_indices()[self.get_indices().index(self._cur_idx) + 1 % len(self.get_indices())]

    def prev_img(self):
        self._cur_idx = self.get_indices()[self.get_indices().index(self._cur_idx) - 1 % len(self.get_indices())]

    def get_city(self):
        return self._cur_city

    def get_cities(self):
        return sorted(list(self._available.keys()))

    def get_idx(self):
        return self._cur_idx

    def get_indices(self):
        return self._available[self.get_city()]

    def _get_stem(self):
        return "{}/{}".format(self.get_city(), self.get_idx())

    def get_video(self):
        return os.path.join(self._vids_dir, self._get_stem() + VIDEO_ENDING)

    def get_image(self):
        return os.path.join(self._img_dir, self._get_stem() + IMG_ENDING)

    def get_label(self):
        return os.path.join(self._label_dir, self._get_stem() + LABEL_ENDING)

    def get_tls(self):
        return os.path.join(self._tl_dir, self._get_stem() + LABEL_ENDING)

    def get_vehicle(self):
        return os.path.join(self._vehicle_dir, self._get_stem() + VEHICLE_ENDING)

    def get_depth(self):
        return DEPTH_FILE


class MainWindow(QMainWindow):

    def __init__(self):
        super().__init__()
        print("discovering")
        self._data = DataLoader()
        print("finished loading")
        print("Reading depths")
        self._depths = DepthHolder(self._data.get_depth())
        print("Finished")
        self._playlist = []  # FIXME 6.3: Replace by QMediaPlaylist?
        self._playlist_index = -1
        self._player = QMediaPlayer()
        self._player.setLoops(10000000)
        self._player.errorOccurred.connect(self._player_error)

        self._label_io = None
        self._draw_stuff = None
        self._redraw_lock = False

        self._status_bar = QStatusBar()
        self.setStatusBar(self._status_bar)
        self.init_toolbar()
        self._main_widget = QWidget()
        self._layout1 = QHBoxLayout(self._main_widget)
        self._layout_tls = QVBoxLayout()
        self._layout1.addLayout(self._layout_tls)
        self._layout_checkboxes_plus_space = QHBoxLayout()
        self._img_widget = QLabel()
        self._img_widget.setScaledContents(True)
        self._pixmap = QPixmap()
        self._img_widget.setPixmap(self._pixmap)
        self._img_widget.setFixedWidth(2048 * SIZE_FACTOR_IMG)
        self._img_widget.setFixedHeight(1024 * SIZE_FACTOR_IMG)
        self._layout_tls.addWidget(self._img_widget)
        self._layout_tls.addLayout(self._layout_checkboxes_plus_space)
        self._layout_checkboxes = QGridLayout()
    # bottom_tool_bar = QToolBar()
        # self.addToolBar(Qt.BottomToolBarArea, bottom_tool_bar)
        self._layout_checkboxes_plus_space.addLayout(self._layout_checkboxes)
        self._layout_global_buttons = QVBoxLayout()
        button_save = QPushButton("Save")
        button_save.clicked.connect(self.onSave)
        self._layout_global_buttons.addWidget(button_save)
        self._layout_checkboxes_plus_space.addLayout(self._layout_global_buttons)
        self._layout_checkboxes_plus_space.addStretch(10)
        self._cb_groups = []
        self._cb_labels = []

        self._tl_buttons = []
        self._img_widget.setPixmap(self._pixmap)

        self._crop_widget = QWidget()
        self._crop_layout = QVBoxLayout(self._crop_widget)
        self._crop_scroll = QScrollArea()
        self._crop_scroll.setWidget(self._crop_widget)
        self._crop_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self._crop_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._crop_scroll.setWidgetResizable(True)
        #self._crop_scroll.setFixedHeight(500)
        #self._crop_layout.setSpacing(10)
        self._crops = []
        self._layout1.addWidget(self._crop_scroll)

        self._video_widget = QVideoWidget()
        self._video_widget.setFixedWidth(2048)

        self._layout1.addWidget(self._video_widget)
        self.setCentralWidget(self._main_widget)
        self._player.setVideoOutput(self._video_widget)

        profile = QWebEngineProfile.defaultProfile()
        profile.setCachePath("/tmp/qcache")
        profile.setPersistentStoragePath("/home/janosovits/.qstorage")
        profile.setPersistentCookiesPolicy(QWebEngineProfile.PersistentCookiesPolicy.ForcePersistentCookies)
        self._web_widget = QWebEngineView()
        self._web_widget.setFixedWidth(500)
        self._web_widget.setUrl("http://maps.google.com/maps?q=&layer=c&cbll=50.93533901728679,6.924794374842279")
        self._mapillary_widget = QWebEngineView()
        self._mapillary_widget.setFixedWidth(500)
        self._mapillary_widget.setUrl("https://www.mapillary.com/app/?lat=47.030860399999995&lng=5.405223299999989&z=17")
        self._web_page = QWebEngineProfile
        self.map_layout = QVBoxLayout()
        self._layout1.addLayout(self.map_layout)
        self.map_layout.addWidget(self._web_widget)
        self.map_layout.addWidget(self._mapillary_widget)

        self._tl_labels = []
        self._city_changed("train/aachen")

    def init_toolbar(self):
        tool_bar = QToolBar()
        self.addToolBar(tool_bar)
        self._cities_combo = QComboBox()
        tool_bar.addWidget(self._cities_combo)
        self._cities_combo.insertItems(0, self._data.get_cities())
        self._cities_combo.activated[int].connect(self.city_changed)

        self._idxs_combo = QComboBox()
        self._idxs_combo.setFixedWidth(200)
        tool_bar.addWidget(self._idxs_combo)
        self._idxs_combo.insertItems(0, self._data.get_indices())
        self._idxs_combo.activated[int].connect(self.image_changed)

        style = self.style()
        icon = QIcon.fromTheme("media-skip-backward-symbolic.svg",
                               style.standardIcon(QStyle.SP_MediaSkipBackward))
        self._previous_action = tool_bar.addAction(icon, "Previous")
        self._previous_action.triggered.connect(self.prev_image)

        icon = QIcon.fromTheme("media-skip-forward-symbolic.svg",
                               style.standardIcon(QStyle.SP_MediaSkipForward))
        self._next_action = tool_bar.addAction(icon, "Next")
        self._next_action.triggered.connect(self.next_image)
        # play_menu.addAction(self._next_action)

        icon = QIcon.fromTheme("media-playback-stop.png",
                               style.standardIcon(QStyle.SP_MediaStop))
        self._stop_action = tool_bar.addAction(icon, "Stop")
        self._stop_action.triggered.connect(self.toggle_play)

        reload = tool_bar.addAction("Reload")
        reload.triggered.connect(self.on_reload)

    def _make_crop_widget(self, tl, tl_idx):
        main = QWidget()
        main.setMaximumHeight(150)
        main.setMaximumWidth(460)
        main.buttons_type = {}
        main.buttons_state = {}
        type_button_widget = QWidget()
        state_button_widget = QWidget()
        hlayout = QHBoxLayout(main)
        vlayout = QVBoxLayout()
        hlayout_top = QHBoxLayout()
        type_layout = QHBoxLayout(type_button_widget)
        state_layout = QHBoxLayout(state_button_widget)
        hlayout_bottom = QHBoxLayout()
        label = QLabel()
        #label.setFixedWidth(50)
        #label.setFixedHeight(100)
        mmin, mmax = pad_box(to_bbox(tl["polygon"]), 10, IMG_SZ)
        crop = self._pixmap_clean.copy(QRect(mmin[0], mmin[1], mmax[0] - mmin[0], mmax[1] - mmin[1])).scaled(QSize(100, 200), Qt.KeepAspectRatio)
        label.setPixmap(crop)
        name = QLabel()
        name.setText(str(tl_idx))
        name.setStyleSheet("font-weight: bold")
        main.spinner = QSpinBox()
        main.spinner.setValue(tl["attributes"]["depth"])
        main.spinner.valueChanged[int].connect(functools.partial(self.on_depth, tl_idx))
        main.spinner.setFixedWidth(50)
        hlayout_top.addWidget(name)
        hlayout.addWidget(label)
        hlayout.addLayout(vlayout)
        vlayout.addLayout(hlayout_top)
        vlayout.addWidget(type_button_widget)
        vlayout.addWidget(state_button_widget)
        vlayout.addLayout(hlayout_bottom)
        hlayout_bottom.addWidget(main.spinner)
        depth_metric = QLabel()
        depth_metric.setText("{:.2f}".format(tl["depth_metric"]))
        hlayout_bottom.addWidget(depth_metric)
        main.del_button = QPushButton("Delete")
        main.del_button.clicked.connect(functools.partial(self.on_delete, tl_idx))
        hlayout_top.addWidget(main.del_button)
        main.visible_button = QCheckBox("Visible")
        if tl["attributes"]["visible"] == "yes":
            main.visible_button.setChecked(True)
        main.visible_button.idx = tl_idx
        main.visible_button.toggled[bool].connect(functools.partial(self.on_visible, tl_idx))
        hlayout_top.addWidget(main.visible_button)
        main.relevant_box = QCheckBox("Relev")
        if tl["attributes"]["relevant"] == "yes":
            main.relevant_box.setChecked(True)
        main.relevant_box.idx = tl_idx
        main.relevant_box.toggled[bool].connect(functools.partial(self.on_relevant, tl_idx))
        hlayout_top.addWidget(main.relevant_box)
        main.lane_relevant_box = QCheckBox("Lane Rel")
        if tl["attributes"]["lane_relevant"] == "yes":
            main.lane_relevant_box.setChecked(True)
        main.lane_relevant_box.idx = tl_idx
        main.lane_relevant_box.toggled[bool].connect(functools.partial(self.on_lane_relevant, tl_idx))
        hlayout_top.addWidget(main.lane_relevant_box)
        for state, s_name in STATE_DICT.items():
            state_button = QRadioButton(state)
            state_button.setStyleSheet("background-color: {}".format(COLOR_DICT[state]))
            state_button.toggled[bool].connect(functools.partial(self.on_state, tl_idx, s_name))
            if s_name == tl["attributes"]["state"]:
                state_button.setChecked(True)
            main.buttons_state[s_name] = state_button
            state_layout.addWidget(state_button)
        for t_name, t_val in TYPE_DICT.items():
            type_button = QRadioButton(t_name)
            type_button.toggled[bool].connect(functools.partial(self.on_type, tl_idx, t_val))
            if t_val == tl["attributes"]["type"]:
                type_button.setChecked(True)
            main.buttons_state[t_val] = type_button
            type_layout.addWidget(type_button)
        return main

    def _clear_crops(self):
        for widget in self._crops:
            self._layout_checkboxes.removeWidget(widget)
            widget.deleteLater()
        del self._crops
        self._crops = []

    def _update_crops(self, tls):
        self._clear_crops()
        for idx, tl in tls.items():
            tl_widget = self._make_crop_widget(tl, idx)
            self._crop_layout.addWidget(tl_widget)
            self._crops.append(tl_widget)

    def _update_video(self):
        self._player.setSource(self._data.get_video())
        self._player.play()

    def _save(self):
        if self._label_io:
            self._label_io.write()
            self._status_bar.showMessage("Wrote {}".format(self._data.get_tls()), 5000)

    def _update_idxs_position(self):
        self._idxs_combo.setCurrentIndex(self._data.get_indices().index(self._data.get_idx()))

    def _update_idxs_list(self):
        self._idxs_combo.clear()
        self._idxs_combo.insertItems(0, self._data.get_indices())
        self._update_idxs_position()

    def _draw_tls(self, tls):
        qp = QPainter(self._pixmap)
        qp.setBrush(Qt.NoBrush)
        qp.setPen(QPen(Qt.red, 5))
        font = QFont()
        font.setPixelSize(25)
        qp.setFont(font)
        for idx, tl in tls.items():
            print(tl["depth_metric"])
            qp.setBrush(Qt.NoBrush)
            qp.setPen(QPen(get_color(tl["attributes"]), 5))
            poly = to_qpolygon(buffer(tl["polygon"], 5))
            qp.drawPolygon(poly)
            box = pad_box(to_bbox(tl["polygon"]), 10, IMG_SZ)
            x = int((box[0][0] + box[1][0]) / 2)
            if box[0][1] > (IMG_SZ[1] / 2):
                y = box[0][1] - 20
            else:
                y = box[1][1] + 10
            #rect = QRect(x, y, 150, 25)
            rect = QRect(x, y, 50, 25)
            brushCol = QColor(Q_COLOR_DICT[tl["attributes"]["state"]])
            brushCol.setAlpha(80)
            qp.setBrush(QBrush(brushCol, Qt.BrushStyle.SolidPattern))
            qp.setPen(Qt.NoPen)
            qp.drawRect(rect)
            qp.setPen(QPen(Qt.black, 5))
            qp.setBrush(Qt.NoBrush)
            #qp.drawText(rect, "{} {:.2f}".format(str(idx), tl["depth_metric"]))
            qp.drawText(rect, "{}".format(str(idx)))
        qp.end()

    def _redraw(self):
        if not self._redraw_lock:
            self._pixmap = self._pixmap_clean.copy()
            self._draw_tls(self._tls)
            self._img_widget.setPixmap(self._pixmap)

    def _update_light_state(self):
        self._tls = self._label_io.get_lights()
        self._redraw()

    def _labels_changed(self):
        depths = self._depths.get_key(self._data._get_stem())
        self._label_io = LabelIO(self._data.get_tls(), depths)
        self._tls = self._label_io.get_lights()
        self._update_crops(self._tls)
        self._redraw()

    def _image_changed(self):
        self._pixmap.load(self._data.get_image())
        self._pixmap_clean = self._pixmap.copy()
        self._labels_changed()
        self._update_idxs_position()
        self._update_video()
        self._web_widget.setUrl(self._get_gmaps())
        self._mapillary_widget.setUrl(self._get_mapillary())

    def _set_image(self, idx):
        self._pre_change()
        self._data.set_idx(idx)
        self._image_changed()

    def _next(self):
        self._pre_change()
        self._data.next_img()
        self._image_changed()

    def _prev(self):
        self._pre_change()
        self._data.prev_img()
        self._image_changed()

    def _pre_change(self):
        self._save()

    def _city_changed(self, city):
        self._pre_change()
        self._data.set_city(city)
        self._update_idxs_list()
        self._image_changed()

    def _toggle_play(self):
        style = self.style()
        if self._player.playbackState() == QMediaPlayer.PlayingState:
            self._player.pause()
            icon = QIcon.fromTheme("media-playback-start.png", style.standardIcon(QStyle.SP_MediaPlay))
            self._stop_action.setIcon(icon)
        else:
            self._player.play()
            icon = QIcon.fromTheme("media-playback-stop.png", style.standardIcon(QStyle.SP_MediaStop))
            self._stop_action.setIcon(icon)

    def _get_gnss(self):
        with open(self._data.get_vehicle(), "r") as f:
            root = json.load(f)
            lat = root["gpsLatitude"]
            lon = root["gpsLongitude"]
            yaw = root["gpsHeading"]
        return lat, lon, yaw

    def _get_gmaps(self):
        lat, lon, yaw = self._get_gnss()
        return "http://maps.google.com/maps?q=&layer=c&cbll={},{}&cbp=12,{},0,0,5".format(lat, lon, yaw)

    def _get_mapillary(self):
        lat, lon, yaw = self._get_gnss()
        return "https://www.mapillary.com/app/?lat={}&lng={}&z=17".format(lat, lon)

    @Slot(QMediaPlayer.Error, str)
    def _player_error(self, error, error_string):
        print(error_string, file=sys.stderr)
        self.show_status_message(error_string)

    @Slot()
    def city_changed(self, city):
        new_city = self._cities_combo.itemText(city)
        self._city_changed(new_city)

    @Slot()
    def image_changed(self, idx):
        self._set_image(self._idxs_combo.itemText(idx))

    @Slot()
    def next_image(self):
        self._next()

    @Slot()
    def prev_image(self):
        self._prev()

    @Slot()
    def toggle_play(self):
        self._toggle_play()

    @Slot()
    def on_type(self, tl_idx, new_type, checked):
        self._label_io.set_type(tl_idx, new_type)
        self._update_light_state()

    @Slot()
    def on_state(self, tl_idx, new_state, checked):
        self._label_io.set_state(tl_idx, new_state)
        self._update_light_state()

    @Slot()
    def on_visible(self, tl_idx, checked):
        self._label_io.set_visible(tl_idx)
        self._update_light_state()

    @Slot()
    def on_relevant(self, tl_idx, checked):
        self._label_io.set_relevant(tl_idx)
        self._update_light_state()

    @Slot()
    def on_lane_relevant(self, tl_idx, checked):
        self._label_io.set_lane_relevant(tl_idx)
        self._update_light_state()

    @Slot()
    def on_delete(self, tl_idx):
        self._label_io.delete_by_idx(tl_idx)
        self._update_light_state()

    @Slot()
    def on_depth(self, tl_idx, depth):
        self._label_io.set_depth(tl_idx, depth)

    @Slot()
    def onCopyStreetviewLinkToClipboard(self):
        QApplication.clipboard().setText("This is text 2 clipboard")

    @Slot()
    def onSave(self):
        self._save()

    @Slot()
    def on_reload(self):
        self._labels_changed()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_win = MainWindow()
    available_geometry = main_win.screen().availableGeometry()
    #main_win.resize(available_geometry.width() - 50,
    #                available_geometry.height() - 100)
    main_win.resize(5100, 1100)
    main_win.show()
    sys.exit(app.exec())