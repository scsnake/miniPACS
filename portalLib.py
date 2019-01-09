# -*- coding: utf-8 -*-

import hashlib
import json
import os
import random
import re
import shutil
import sys

import requests
import urllib3

urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
from bs4 import BeautifulSoup as bs

from functools import wraps
import inspect
from win32func import WM_COPYDATA_Listener, Send_WM_COPYDATA

# from multiprocessing import Pool
import atexit
from concurrent.futures.thread import _python_exit
from functools import partial
import threading


def initializer(func):
    """
    Automatically assigns the parameters.

    >>> class process:
    ...     @initializer
    ...     def __init__(self, cmd, reachable=False, user='root'):
    ...         pass
    >>> p = process('halt', True)
    >>> p.cmd, p.reachable, p.user
    ('halt', True, 'root')
    """
    names, varargs, keywords, defaults = inspect.getargspec(func)

    @wraps(func)
    def wrapper(self, *args, **kargs):
        for name, arg in list(zip(names[1:], args)) + list(kargs.items()):
            setattr(self, name, arg)

        for name, default in zip(reversed(names), reversed(defaults)):
            if not hasattr(self, name):
                setattr(self, name, default)

        func(self, *args, **kargs)

    return wrapper


class PacsMobile:
    def __init__(self, loginId='', loginPw=''):
        self.loginId = loginId
        self.loginPw = loginPw
        self.conn = requests.session()
        self.conn.verify = False
        self.url = r'https://pacsmobilenew.ntuh.gov.tw'
        self.host = r'pacsmobilenew.ntuh.gov.tw'
        self.resolution = r'4096 * 5120'
        self.ua = r'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0;  rv:11.0) like Gecko'
        self.ua_header = {'User-Agent': self.ua,
                          'Accept': r'text/html, application/xhtml+xml, */*',
                          'Accept-Language': 'zh-TW',
                          'Content-Type': r'application/x-www-form-urlencoded',
                          'Accept-Encoding': 'gzip, deflate',
                          'DNT': '1',
                          'Connection': 'Keep-Alive'}
        r = self.conn.get(self.url, headers=self.ua_header)
        r.raise_for_status()
        if loginId != '' and loginPw != '':
            self.login()

    def login(self):
        headers = {'Referer': self.url,
                   'Host': self.host}
        data = {'hash': '',
                'user': self.loginId,
                'password': self.loginPw,
                'domain': 'Impax',
                'resolution': self.resolution}
        r = self.conn.post(self.url, data=data, headers=self.ua_header.update(headers))
        r.raise_for_status()
        print(r.headers)


class Portal:
    def __init__(self, loginId='', loginPw='', hospitalCode='T0', noLogin=False):
        self.hospital_code = hospitalCode
        self.loginId = loginId
        self.loginPw = loginPw
        self.sid = ''
        self.sid_pool = []
        self.loginStatus = False
        self.post_state_var = ['__VIEWSTATE', '__EVENTVALIDATION']
        self.post_state = {var: '' for var in self.post_state_var}
        self.cookie_state_var = ['ASP.NET_SessionId']
        self.cookie_state = {var: '' for var in self.cookie_state_var}
        self.conn = None
        self.ua = r'Mozilla/5.0 (compatible, MSIE 11, Windows NT 6.3; Trident/7.0;  rv:11.0) like Gecko'
        self.ua_header = {'User-Agent': self.ua}
        self.url = 'http://portal.ntuh.gov.tw'

        if loginId != '' and loginPw != '' and not noLogin:
            self.login()

    def refresh_value(self, url, data={}, headers={}):

        if bool(data):
            r = requests.post(url, data=data, headers=headers.update(self.ua_header))
        else:
            r = requests.get(url, headers=headers.update(self.ua_header))

        r.raise_for_status()
        c = r.cookies
        for var in self.cookie_state_var:
            if var in c:
                t = c[var]
                if t:
                    self.cookie_state[var] = t

        s = bs(r.text, 'html.parser')
        for var in self.post_state_var:
            t = s.find(id=var)
            if t:
                self.post_state[var] = t['value'] if t.has_attr('value') else t.string

    def md5(self, s):
        return hashlib.md5(s.encode('utf-8)')).hexdigest()

    def append_post(self, data={}, var=[]):
        if len(var) == 0:
            var = self.post_state_var

        for v in var:
            data[v] = self.post_state.get(v, '')

        return data

    def append_cookies(self, cookies={}, var=''):
        if var is list:
            for v in var:
                cookies[v] = self.cookie_state.get(v, '')
        elif var != '':
            cookies[var] = self.cookie_state.get(var, '')
        else:
            for v in self.cookie_state_var:
                cookies[v] = self.cookie_state.get(v, '')
        return cookies

    def login(self):
        url = r'http://sess.ntuh.gov.tw/SessionService/SessionService.asmx'
        header = {'SOAPAction': "http://140.112.3.247/General/SessionService.asmx/NewSession",
                  'Host': 'sess.ntuh.gov.tw',
                  'Content-Type': 'text/xml; charset=utf-8'}
        self.loginPwMd5 = self.md5(self.loginPw)
        randomIp = str(random.randint(1, 254))
        data = r'<?xml version="1.0" encoding="utf-8"?><soap:Envelope xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/" xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance" xmlns:xsd="http://www.w3.org/2001/XMLSchema"><soap:Body><NewSession xmlns="http://140.112.3.247/General/SessionService.asmx"><UserID>{}</UserID><password>{}</password><RemoteIP>172.17.149.{}</RemoteIP><strHospital>{}</strHospital></NewSession></soap:Body></soap:Envelope>'.format(
            self.loginId,
            self.loginPwMd5,
            randomIp,
            self.hospital_code
        )

        r = requests.post(url, data=data, headers=header)
        r.raise_for_status()

        result = re.search(r'[A-Z\d]{32,}', r.text, re.I)
        if result:
            self.sid = result.group(0)
            self.sid_pool.append(self.sid)
            self.conn = requests.session()
            self.conn.verify = False
            self.conn.get(r'http://portal.ntuh.gov.tw/General/NewRedirect.aspx?SESSION=' + self.sid)
            self.loginStatus = True
            return self.sid
        else:
            return ''

    def pacsMobileNew(self, accNo, chartNo):
        url = r'https://pacsmobilenew.ntuh.gov.tw/?user={}&password={}&tab=Display&PatientID={}&AccessionNumber={}'.format(
            self.loginId,
            self.loginPw,
            chartNo,
            accNo
        )
        headers = {
            'Referer': r'http://ihisaw.ntuh.gov.tw/WebApplication/OtherIndependentProj/PatientBasicInfoEdit/PACS/MobileScreen.aspx',
            'Host': 'pacsmobilenew.ntuh.gov.tw'}

        self.conn = requests.session()
        self.conn.verify=False
        r = self.conn.get(url, verify=False)
        # r = self.conn.get(url, headers = headers, verify=False)
        r.raise_for_status()
        # print(r.headers)

    def pacsMobileNewInfo(self, accNo, chartNo):
        self.pacsMobileNew(accNo, chartNo)

        url = r'https://pacsmobilenew.ntuh.gov.tw/wado/?v=3.0.SU6.23&requestType=STUDY&contentType=text/javascript&maxResults=500&PatientID={}&AccessionNumber={}&ae=local&IssuerOfPatientID=&groupByIssuer=*&suppressReportFlags=PRELIMINARY&language=zh_TW&theme=theme'.format(
            chartNo, accNo
        )
        headers = {'Referer': 'https://pacsmobilenew.ntuh.gov.tw/',
                   'Host': 'pacsmobilenew.ntuh.gov.tw'}

        r = self.conn.get(url, headers=self.ua_header.update(headers), verify=False)
        r.raise_for_status()

        try:
            info = json.loads(r.text)
        except:
            return None

        studyUID = info['children'][0]['children'][1]['studyUID']
        ver = info['children'][0]['children'][1]['ver']
        count = info['children'][0]['children'][1]['NumberOfStudyRelatedInstances']

        url = r'https://pacsmobilenew.ntuh.gov.tw/wado/?v=3.0.SU6.23&requestType=IMAGE&contentType=text/javascript&regroup=*&studyUID={}&Position=0&Count=2&extraDicomAttributes=ImageOrientationPatient\ImagePositionPatient\PixelSpacing\FrameOfReferenceUID&fromHeader=false&suppressReportFlags=PRELIMINARY&ver={}&ae=local&gsps=Impax%20Presentation&language=zh_TW&theme=theme'.format(
            studyUID, ver)

        r = self.conn.get(url, headers=self.ua_header.update(headers), verify=False)
        r.raise_for_status()

        try:
            info = json.loads(r.text)
            result = info.get('children', [])
            return DicomModel('patient', result[0]) if len(result) > 0 else None
            # return info['children'][0]
        except:
            return None

    def pacsMobileNewTrans(self, study_uid, series_uid, object_uid, baseUrl):
        return r'{}/wado/?requestType=XERO&studyUID={}&seriesUID={}&language=zh_TW&objectUID={}&ae=local&v=3.0.SU6.23'.format(
            baseUrl, study_uid, series_uid, object_uid)

    def pacsMobileNewDownloadJpeg(self, accNo, chartNo, folder='', url='', fileName='', picWidth='', retry=3,
                                  fileSize=100,
                                  first=True, callback=None, flatten=False, callback2=None):
        '''

        :param accNo:
        :param chartNo:
        :param folder:
        :param url:
        :param fileName:
        :param picWidth:
        :param retry:
        :param fileSize:
        :param first:
        :param callback: callback function when each image downloaded
        :param flatten:
        :param callback2: callback function when all images downloading, for image count
        :return:
        '''
        baseUrl, host = self.pacsWebServer(accNo)

        if folder == '':
            folder = os.path.dirname(os.path.abspath(__file__))

        if url == '':
            info = self.pacsMobileNewInfo(accNo, chartNo)
            if flatten:
                image_count = 0
                for i, image in enumerate(info.image):
                    # print('Downloading: %s\n%s (%s)\n%s\n%s\n' % \
                    #       (image.patient.PatientID, image.study.StudyDescription, image.study.AccessionNumber,
                    #        image.series.SeriesDescription, i))
                    image_name = accNo + ' ' + chartNo + ' ' + str(i + 1) + '.jpeg'
                    path = os.path.join(folder, image_name)
                    if os.path.exists(path) and os.path.getsize(path) > fileSize * 1000:
                        print('{} already downloaded.'.format(image_name))
                    else:
                        self.pacsMobileNewDownloadJpeg(accNo, chartNo, folder,
                                                       self.pacsMobileNewTrans(image.study.studyUID,
                                                                               image.series.seriesUID,
                                                                               image.objectUID,
                                                                               baseUrl),
                                                       image_name,
                                                       picWidth, retry, fileSize, first, callback)
                    image_count += 1

                if callback2:
                    callback2(accNo, chartNo, image_count)


            else:
                study = info.study[0]
                study_dir = os.path.join(folder,
                                         re.sub(r'[^a-zA-Z0-9]', '_',
                                                study.DateF.replace('/', '') + ' ' + study.Description))
                for series in study.series:
                    uid = series.UID
                    if not uid:
                        continue
                    series_dir = os.path.join(study_dir, series.SeriesNumber)
                    if not os.path.exists(series_dir):
                        os.makedirs(series_dir)
                    for i, image in enumerate(series.image):
                        image_name = accNo + ' ' + chartNo + ' ' + str(i + 1) + '.jpeg'
                        path = os.path.join(series_dir, image_name)
                        if os.path.exists(path) and os.path.getsize(path) > fileSize * 1000:
                            print('{} already downloaded.'.format(image_name))
                        else:
                            self.pacsMobileNewDownloadJpeg(accNo, chartNo, series_dir,
                                                           self.pacsMobileNewTrans(image.study.studyUID,
                                                                                   image.series.seriesUID,
                                                                                   image.objectUID,
                                                                                   baseUrl),
                                                           image_name,
                                                           picWidth, retry, fileSize, first, callback)

            return
        elif url is list:
            for i, u in enumerate(url):
                self.pacsMobileNewDownloadJpeg(accNo, chartNo, folder, u,
                                               accNo + ' ' + chartNo + ' ' + str(i + 1) + '.jpeg',
                                               picWidth, retry, fileSize, first, callback)
            return
        picWidth = str(picWidth if picWidth else 4000)

        # if first:
        #     if 'hwpacs.ylh.gov.tw' in baseUrl:
        #         if not 'User=' in url:
        #             url += '&User=' + self.loginId
        #         if not 'Password=' in url:
        #             url += '&Password=' + self.md5(self.loginPw)
        #         if not 'columns=' in url:
        #             url += '&columns=' + picWidth
        #         elif picWidth:
        #             url = re.sub(r'(?<=columns=)[^&]+', picWidth, url)
        #     else:
        #         if 'pacsmobile' in baseUrl:
        #             if not 'user=' in url:
        #                 url += '&user=' + self.loginId
        #             if not 'password=' in url:
        #                 url += '&password=' + self.loginPw
        #         if not 'rows=' in url:
        #             url += '&rows=' + picWidth
        #         elif picWidth:
        #             url = re.sub(r'(?<=rows=)[^&]+', picWidth, url)

        if not os.path.exists(folder):
            os.makedirs(folder)
        path = os.path.join(folder, fileName)
        if os.path.exists(path):
            if os.path.getsize(path) > fileSize * 1000:
                print('{} already downloaded.'.format(fileName))
                return
            else:
                os.remove(path)
        print('Downloading {} ...'.format(fileName))
        headers = {'Referer': baseUrl, 'Host': host}

        r = self.conn.get(url, headers=self.ua_header.update(headers), stream=True, verify=False)
        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            return

        if int(r.headers['Content-Length']) > int(fileSize):
            if fileName == '':
                fileName = accNo + ' ' + chartNo + ' 1.jpeg'

            with open(path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

            if callback is not None:
                callback(accNo=accNo, chartNo=chartNo, folder=folder, fileName=fileName, path=path, raw=r.raw)

            return
        else:
            retry -= 1
            if retry < -1:
                return -1
            else:
                return self.pacsMobileNewDownloadJpeg(accNo, chartNo, folder, url, fileName, picWidth, retry, fileSize,
                                                      False,
                                                      callback, flatten)

    def pacsWebInfoYLH(self, accNo, chartNo='', count=1500):
        pass

    def pacsWebInfo(self, accNo, chartNo='', count=1500):
        baseUrl, host = self.pacsWebServer(accNo)

        if host == 'hwpacs.ylh.gov.tw':
            return self.pacsWebInfoYLH(accNo, chartNo, count)
        elif 'pacsmobile' in baseUrl:
            url = baseUrl + '/wado/?v=2.0.SU3.3&requestType=STUDY&contentType=text/javascript&maxResults=250&PatientID=%s&AccessionNumber=%s&ae=local&IssuerOfPatientID=&groupByIssuer=*&language=zh_TW&user=%s&password=%s' % \
                            (chartNo, accNo, self.loginId, self.loginPw)
            referer = baseUrl + '/?&tab=Display&PatientID=%s&AccessionNumber=%s&theme=epr' % (chartNo, accNo)

            headers = {'Referer': referer, 'Host': host}
            self.conn = requests.session()
            self.conn.verify=False
            self.conn.headers = self.ua_header
            r = self.conn.get(url, headers=headers)
            r.raise_for_status()
        elif re.search(r'xero-\d+', baseUrl, re.I):
            url = 'http://pacswide.ntuh.gov.tw:8080/'
            headers = {'Host': 'pacswide.ntuh.gov.tw:8080'}
            self.conn = requests.session()
            self.conn.verify = False
            self.conn.headers = self.ua_header
            r = self.conn.get(url, headers=headers, allow_redirects=False)
            r.raise_for_status()

            url = baseUrl + '/'
            r = self.conn.get(url, headers=headers, allow_redirects=False)
            r.raise_for_status()

            url = baseUrl + '/?user=xero01&password=xero01&tab=Display&PatientID=%s&AccessionNumber=%s&theme=epr' % (
                chartNo, accNo)
            headers = {'Referer': 'http://pacswide.ntuh.gov.tw:8080/kQuery.php',
                       'Host': host}
            r = self.conn.get(url, headers=headers, allow_redirects=False)
            r.raise_for_status()

            url = baseUrl + '/wado/?v=2.0.SU3.3&requestType=STUDY&contentType=text/javascript&maxResults=250&PatientID=%s&AccessionNumber=%s&ae=local&IssuerOfPatientID=&groupByIssuer=*&language=zh_TW&user=%s&password=%s' % \
                            (chartNo, accNo, self.loginId, self.loginPw)
            headers = {
                'Referer': baseUrl + '/?&tab=Display&PatientID=%s&AccessionNumber=%s&theme=epr' % (chartNo, accNo),
                'Host': host}
            r = self.conn.get(url, headers=headers, allow_redirects=False,
                              cookie=self.conn.cookie.update({'xero-viewport-size': '1184/1152'}))
            r.raise_for_status()

        try:
            info = json.loads(r.text)
        except:
            return None

        studyUID = info['children'][0]['children'][1]['studyUID']
        ver = info['children'][0]['children'][1]['ver']
        count = info['children'][0]['children'][1]['NumberOfStudyRelatedInstances']

        url = baseUrl + '/wado/?v=2.0.SU3.3&requestType=IMAGE&contentType=text/javascript&regroup=*&studyUID=%s&Position=0&Count=%s&ver=%s&ae=local&language=zh_TW' % \
                        (studyUID, count, ver)
        if 'pacsmobile' in baseUrl:
            url += '&user=%s&password=%s' % (self.loginId, self.loginPw)

        headers = {'Referer': baseUrl + '/?&tab=Display&PatientID=%s&AccessionNumber=%s&theme=epr' % (chartNo, accNo),
                   'Host': host}
        r = self.conn.get(url, headers=headers)
        r.raise_for_status()

        try:
            info = json.loads(r.text)
            result = info.get('children', [])
            return DicomModel('patient', result[0]) if len(result) > 0 else None
            # return info['children'][0]
        except:
            return None

    def pacsWebInfoFlatten(self, study_info):
        if study_info is None:
            return []

        ret = []
        for series in study_info.get('children', []):
            uid = series.get('seriesUID', '')
            if uid == '':
                continue
            for object in series.get('children', []):
                o = object.get('children', [])
                if len(o) > 0 and o[0].get('objectUID', '') != '':
                    ret.append(o[0])
        return ret

    def pacsWebDownloadJpeg(self, accNo, chartNo, folder='', url='', fileName='', picWidth='', retry=3, fileSize=100,
                            first=True, callback=None, flatten=False):

        baseUrl, host = self.pacsWebServer(accNo)

        if folder == '':
            folder = os.path.dirname(os.path.abspath(__file__))

        if url == '':
            info = self.pacsWebInfo(accNo)
            if flatten:
                for i, image in enumerate(info.image):
                    print('Downloading: %s\n%s (%s)\n%s\n%s\n' % \
                          (image.patient.PatientID, image.study.StudyDescription, image.study.AccessionNumber,
                           image.series.SeriesDescription, i))
                    self.pacsWebDownloadJpeg(accNo, chartNo, folder,
                                             self.pacsWebUrlTrans(image.url, baseUrl, host),
                                             accNo + ' ' + chartNo + ' ' + str(i + 1) + '.jpeg',
                                             picWidth, retry, fileSize, first, callback)
            else:
                study = info.study[0]
                study_dir = os.path.join(folder,
                                         re.sub(r'[^a-zA-Z0-9]', '_',
                                                study.DateF.replace('/', '') + ' ' + study.Description))
                for series in study.series:
                    uid = series.UID
                    if not uid:
                        continue
                    series_dir = os.path.join(study_dir, series.SeriesNumber)
                    if not os.path.exists(series_dir):
                        os.makedirs(series_dir)
                    for i, image in enumerate(series.image):
                        self.pacsWebDownloadJpeg(accNo, chartNo, series_dir,
                                                 self.pacsWebUrlTrans(image.url, baseUrl, host),
                                                 accNo + ' ' + chartNo + ' ' + str(i + 1) + '.jpeg',
                                                 picWidth, retry, fileSize, first, callback)

            return
        elif url is list:
            for i, u in enumerate(url):
                self.pacsWebDownloadJpeg(accNo, chartNo, folder, u,
                                         accNo + ' ' + chartNo + ' ' + str(i + 1) + '.jpeg',
                                         picWidth, retry, fileSize, first, callback)
            return
        picWidth = str(picWidth if picWidth else 4000)

        if first:
            if 'hwpacs.ylh.gov.tw' in baseUrl:
                if not 'User=' in url:
                    url += '&User=' + self.loginId
                if not 'Password=' in url:
                    url += '&Password=' + self.md5(self.loginPw)
                if not 'columns=' in url:
                    url += '&columns=' + picWidth
                elif picWidth:
                    url = re.sub(r'(?<=columns=)[^&]+', picWidth, url)
            else:
                if 'pacsmobile' in baseUrl:
                    if not 'user=' in url:
                        url += '&user=' + self.loginId
                    if not 'password=' in url:
                        url += '&password=' + self.loginPw
                if not 'rows=' in url:
                    url += '&rows=' + picWidth
                elif picWidth:
                    url = re.sub(r'(?<=rows=)[^&]+', picWidth, url)

        if 'hwpacs.ylh.gov.tw' in baseUrl:
            headers = {
                'Referer': baseUrl + '/html5/ShowImage.html?User=%s&Password=%s&patientID=%s&accessionNumber=%s' % \
                                     (self.loginId,
                                      self.md5(self.loginPw),
                                      chartNo, accNo),
                'Host': host}
        else:
            headers = {'Referer': baseUrl + '/?&tab=Display&PatientID=%s&AccessionNumber=%s&theme=epr' % \
                                            (chartNo, accNo),
                       'Host': host}

        r = self.conn.get(url, headers=headers, stream=True)
        try:
            r.raise_for_status()
        except Exception as e:
            print(e)
            return

        if int(r.headers['Content-Length']) > int(fileSize):
            if fileName == '':
                fileName = accNo + ' ' + chartNo + ' 1.jpeg'
            if not os.path.exists(folder):
                os.makedirs(folder)
            path = os.path.join(folder, fileName)
            if os.path.exists(path):
                os.remove(path)
            with open(path, 'wb') as f:
                r.raw.decode_content = True
                shutil.copyfileobj(r.raw, f)

            if callback is not None:
                callback(accNo=accNo, chartNo=chartNo, folder=folder, fileName=fileName, path=path, raw=r.raw)

            return
        else:
            retry -= 1
            if retry < -1:
                return -1
            else:
                return self.pacsWebDownloadJpeg(accNo, chartNo, folder, url, fileName, picWidth, retry, fileSize, False,
                                                callback, flatten)

    def pacsWebUrlTrans(self, url, baseUrl, host):

        studyUID = re.search('(?<=studyUID=)[^&]+', url, re.I | re.DOTALL)
        seriesUID = re.search('(?<=seriesUID=)[^&]+', url, re.I | re.DOTALL)
        objectUID = re.search('(?<=objectUID=)[^&]+', url, re.I | re.DOTALL)

        if 'pacsmobilenew' in baseUrl:
            return r'{}/wado/?requestType=XERO{}{}&language=zh_TW{}&ae=local&v=3.0.SU6.23'.format(
                '&studyUID=' + studyUID.group(0) if studyUID else '',
                '&seriesUID=' + seriesUID.group(0) if seriesUID else '',
                '&objectUID=' + objectUID.group(0) if objectUID else ''
            )
        else:
            return baseUrl + '/wado/?v=2.0.SU3.3&requestType=WADO%s%s%s&language=zh_TW&ae=local' % \
                             ('&studyUID=' + studyUID.group(0) if studyUID else '',
                              '&seriesUID=' + seriesUID.group(0) if seriesUID else '',
                              '&objectUID=' + objectUID.group(0) if objectUID else ''
                              )

    def pacsWebServer(self, AccNo):
        if re.search(r'^Y', AccNo, re.I):
            return 'http://hwpacs.ylh.gov.tw', 'hwpacs.ylh.gov.tw'

        hospitalCode = re.search(r'^T(\d)', AccNo, re.I)
        if hospitalCode and (hospitalCode.group(1) == '4' or str(AccNo).startswith('H')):
            return 'http://xero-01.hch.gov.tw', 'xero-01.hch.gov.tw'
        else:
            return 'https://pacsmobilenew.ntuh.gov.tw', 'pacsmobilenew.ntuh.gov.tw'


class DicomModel:
    levels = ('patient', 'study', 'series', 'image')

    def __init__(self, level, info, parent=None):
        # input first result as json['children'][0] if any

        level = str(level).lower()
        assert level in DicomModel.levels
        self.info = info
        self.level = level
        self.level_n = DicomModel.levels.index(level)

        if self.level_n > 0:
            parent_level = DicomModel.levels[self.level_n - 1]
            setattr(self, parent_level, parent)
            self.parent = parent

        if self.level_n < 3 and 'children' in info:
            child_level = DicomModel.levels[self.level_n + 1]
            self.children = [DicomModel(child_level, i, self) for i in info['children'] if
                             i.get('tagName', '') == child_level]
            setattr(self, child_level, self.children)

    def __getattr__(self, item):
        item = str(item)
        if item in DicomModel.levels and item != self.level:
            if DicomModel.levels.index(item) < self.level_n:
                return getattr(self.parent, item)
            else:
                ret = []
                for c in self.children:
                    ret += getattr(c, item)
                return ret
        elif item in self.info:
            return self.info[item]
        elif (self.level + item) in self.info:
            return self.info[(self.level + item)]
        elif (self.level.title() + item) in self.info:
            return self.info[(self.level.title() + item)]

    def __del__(self):
        children = self.children
        if children:
            for ch in children:
                ch.__del__()

        try:
            self.parent.children.remove(self)
        except:
            pass


class Main:
    def __init__(self):
        self.bridge_hwnd = 0
        self.miniPACS_hwnd = 0
        self.getAllJpeg_hwnd = 0
        self.dwData = 17
        self.portal = None
        # self.executor = ThreadPoolExecutor(max_workers=10)
        self.exit = False
        self.downloading = []
        self.WM_COPYDATA_Listener = WM_COPYDATA_Listener(receiver=self.listener,
                                                         title='WM_COPYDATA_Listener_portalLib',
                                                         useThread=False)

    def listener(self, **kwargs):
        try:
            if kwargs['dwData'] != self.dwData:
                return

            json_data = json.loads(kwargs['lpData'])

            if 'setBridgeHwnd' in json_data:
                self.bridge_hwnd = int(json_data['setBridgeHwnd'])
            elif 'portalLogin' in json_data:
                loginId = json_data['loginId']
                loginPw = json_data['loginPw']
                if self.portal is None:
                    self.portal = Portal(loginId, loginPw, noLogin=True)

            elif 'downloadJpeg' in json_data:
                accNo = json_data['accNo']
                chartNo = json_data['chartNo']
                folder = json_data['folder']
                index = json_data['index']
                if accNo in self.downloading:
                    return
                else:
                    self.downloading.append(accNo)
                # self.portal.pacsMobileNewDownloadJpeg(accNo, chartNo)
                threading.Thread(target=self.portal.pacsMobileNewDownloadJpeg, args=(accNo, chartNo, folder),
                                 kwargs={'flatten': True,
                                         'callback2': partial(self.sendStudyInfo, index=index)}).start()
                r = Send_WM_COPYDATA(self.getAllJpeg_hwnd, json.dumps({'received': 1}), self.dwData)
                print(r)
                # print('Downloading: {} {} in {} ...'.format(
                #     accNo, chartNo, folder
                # ))
            elif 'miniPACS_hwnd' in json_data:
                self.miniPACS_hwnd = json_data['miniPACS_hwnd']
            elif 'getAllJpegHwnd' in json_data:
                self.getAllJpeg_hwnd = json_data['getAllJpegHwnd']
            elif 'exit' in json_data:
                atexit.unregister(_python_exit)
                self.executor.shutdown = lambda wait: None
                self.exit = True
                sys.exit(0)



        except Exception as e:
            print(e)
            return

    def sendStudyInfo(self, accNo, chartNo, image_count, index):
        msg = {'study_data': 1}
        msg[index] = {'AccNo': accNo,
                      'ChartNo': chartNo,
                      'expected_image_count': [{}]}
        msg[index]['expected_image_count'][0][accNo] = image_count
        Send_WM_COPYDATA(self.miniPACS_hwnd, json.dumps(msg), self.dwData)

    def send_WM(self, msg):
        Send_WM_COPYDATA(self.bridge_hwnd, json.dumps(msg), self.dwData)


if __name__ == '__main__':
    p = Portal('100860', r'4RFV5tgb')
    p.pacsMobileNewDownloadJpeg('T0185543242', '6558738')
    a = 1
    # while not main.exit:
    #     pass
