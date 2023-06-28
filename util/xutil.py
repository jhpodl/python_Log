# --- import------------------------------
import os
import re
import json
import time
from typing import Set
import uuid
import platform
import operator
import gzip
import shutil
import random
import calendar
import mmap
import tarfile
import time
import linecache as lc

import multiprocessing
import orjson
from panel import extension

from logging import exception
from genericpath import isdir, isfile
from tokenize import group
from datetime import datetime
from types import GeneratorType
from collections import OrderedDict
from itertools import product
from multiprocessing import Pool
from collections import OrderedDict
from zipfile import ZipFile
from string import hexdigits
from time import sleep
from timeit import repeat
from contextlib import contextmanager
import numpy as np
import numexpr as ne
import pandas as pd
import pyarrow as pa
from pyarrow import json as pajson
import pyarrow.parquet as pq
import requests

os_name = 'WIN' if platform.system() == 'Windows' else 'NIX'
os_nix_splitter = '/'
os_win_splitter = '\\'
ub = '_'
os_path_splitter = os_win_splitter if os_name == 'WIN' else os_nix_splitter

idx_first = 0
idx_second = 1
idx_last = -1  # first from last

_KILO_ = 1024
_MEGA_ = _KILO_ * 1024
_GIGA_ = _MEGA_ * 1024
_TERA_ = _GIGA_ * 1024

_OS_ = platform.system()
_PATH_SPLITTER_ = '\\' if _OS_ == 'Windows' else '/'
_void = '∅_'

def get_moudle_name():
    return 'name'

# functional util ####################### 나중에 모듈로 다시 만들어야 함
# --- 타입관련
def fn_(x): x
def just(v, *vs): return v

def is_it(x): return x is not None
def is_none(x): return x is None
# Boolean Types
def is_bool(x): return type(x) is bool

## Numeric Types
def is_int(x): return type(x) is int
def is_float(x): return type(x) is float
def is_complex(x): return type(x) is complex
def is_num(x): return is_float(x) or is_int(x) or is_complex(x)

## Sequence Types
def is_str(x): return type(x) is str
def is_list(x): return type(x) is list
def is_tuple(x): return type(x) is tuple
def is_seq(x): return is_list(x) or is_tuple(x) or is_tuple(x)

## Mapping Type
def is_dict(x): return type(x) is dict

## Set Type
def is_set(x): return type(x) is set

## Function Type
def is_fun(x): return type(x) is type(fn_)
def is_gen(x): return type(x) is GeneratorType


def is_range(x): return type(x) is range
def is_regex(x) : return type(x) is re.Pattern
def is_iterable(x): return is_it(x) and (is_str(x) or is_list(x) or is_tuple(x) or is_gen(x) or is_range(x) or is_set(x))

def is_iter_but_str(x): return is_it(x) and (is_list(x) or is_tuple(x) or is_gen(x) or is_range(x))
def is_collect_but_gen(x): return is_it(x) and (is_list(x) or is_tuple(x) or is_dict(x))

# null 이면 -1, 기타 len(x)
def is_it_len(x): -1 if is_none(x) else len(x)

# 자기자신을 리턴함
def identity(x): return x

def functor(x): 
    def mk_fun(*xn):
        return x
    return x if is_fun(x) else mk_fun

def predicator(f):
    ftype = '' 
    if is_regex(f):
        ftype = 'regex'
    elif is_fun(f):
        ftype = 'function'
    else:
        ftype = 'None'
    
    def predicator_(x):
        if ftype == 'function':
            return True if f(x) is not False else False
        elif ftype == 'regex':
            return False if re.match(f, x) is None else True
        else: 
            return True if f == x else False
        
    return predicator_

def is_true(x): return x is True
def is_false(x): return x is False
def is_not_false(x): return x is not False
def is_even(x): return x % 2 == 0
def is_odd(x): return x % 2 == 1
def is_not_for(f, v): return predicator(f)(v) is not True
def is_truethy(x): return is_it(x) and is_not_false(x)
def is_falsy(x): return is_none(x) or is_false(x)
def is_empty_str(x) : return is_none(x) or (is_str(x) is False) or len(x) == 0
def is_not_empty_str(x) : return is_it(x) and is_str(x) and len(x) > 0

def type_(o):
    if is_str(o):
        return 'string'
    elif is_list(o):
        return 'list'
    elif is_dict(o):
        return 'dict'
    elif is_gen(o):
        return 'gen'
    elif is_range(o):
        return 'range'
    elif is_fun(o):
        return 'function'
    elif is_num(o):
        return 'number'
    elif is_float(o):
        return 'float'
    elif is_none(o):
        return 'none'
    elif is_complex(o):
        return 'complex'


def to_float(x, r_cnt=10, default=0.0):
    try: 
        fv = float(x)
    except:
        fv = default
    finally:
        return round(fv,  r_cnt)


def to_int(x, default=0):
    try:
        iv = int(x)
    except:
        iv = default
    return iv

def of(k, x):
    r = None
    try:
        r = x[k]
    except:
        r = None
    finally:
        return r
    
def get(k, x, alternative):
    r = None
    try:
        r = x[k]
    except:
        r = alternative
    finally:
        return r
    
    
def getter(k, f=None):
    indexer = make_indexer(k)
    mf = identity if f is None else functor(f)
    
    def getter_(x):
        v = indexer(x)
        return mf(v)
    return getter_

def extract_from(*ks):
    def extract(x):
        r = {}
        for k in ks:
            v = x[k]
            r[k] = v
        return r
    return extract

# 스트림 함수 (stream, iteralbe)을 받아서 스트림으로 변환하여 yield
def stream(*xn):
    for x in xn:
        if is_iter_but_str(x):
            for item in x:
                yield item
        elif is_dict(x):
            for k, v in x.items():
                yield {k: v}
        elif is_range(x):
            for item in x:
                yield item
        elif is_gen(x):
            for _ in x:
                yield _
        else:
            yield x

def deep_stream(*xn):
    for x in stream(*xn):
        if is_iter_but_str(x):
            for _ in deep_stream(x):
                yield _
        else:
            yield x
            
def deep_copy(o):
    t = type_(o)
    r = None
    if t == 'dict':
        r = {}
    elif t == 'list':
        r = []
    else:
        r = o
    
    if t == 'list':
        for x in o:
            r.append(deep_copy(x))
    elif t == 'dict':
        for k, v in o.items():
            r[k] = deep_copy(v)
    return r

# 판별함수 모든 것을 만족해야 참리턴
def is_all(f, *xn):
    pf = predicator(f)
    for x in stream(*xn):
        if pf(x) is False:
            return False
    return"true"

# 판별함수 범위(vs)와 일치하면 참 sql in 과 같은 함수
def is_in(f, *xn):
    pf = predicator(f)
    for x in stream(*xn):
        if pf(x) is True:
            return True
    return False

# 판별함수 범위(vs)와 일치하면 거짓 sql not in 과 같은 함수
def is_not_in(f, *vs):
    pf = predicator(f)
    for s in stream(*vs):
        if pf(s) is not False:
            return True
    return True

# 객체의 처음 원소를 리턴
def first_of(obj):
    idx = 0
    first_item = _void
    if is_str(obj):
        return v[0:1] if len(obj) > 1 else _void
    elif is_dict(obj):
        for k, v in obj.items():
            return {k: v}
        return _void
    elif is_gen(obj):
        generated = False
        for o in obj:
            generated ="true"
            r = o
            break
        return r if generated else _void
    elif is_iter_but_str(obj):
        return obj[0] if len(obj) > 0 else _void
    else:
        return obj

def filter_(f):
    pf = predicator(f)

    def filter__(*xn):
        for x in stream(*xn):
            r = False
            try:
                r = pf(x)
            except:
                r = False
            finally:
                if r != False:
                    yield x
    return filter__

# map closure
def map_(f):
    fn = functor(f)
    def map__(*xn):
        for x in stream(*xn):
            r = None
            try:
                r = fn(x)
            except:
                r = None
            finally:
                yield r
    return map__


def reject_(f):
    pf = predicator(f)
    def reject__(*xn):
        for x in stream(*xn):
            r = False
            try:
                r = pf(x)
            except:
                r = False
            finally:
                if r != False:
                    yield x
    return reject__

def make_indexer(indexer):
    if is_fun(indexer):
        return functor(indexer)
    elif is_str(indexer) or is_int(indexer):
        def index_(x):
            r = None
            try:
                r = x[indexer]
            except:
                r = indexer
            finally:
                return r
        return index_
    else:
        return identity


#스트림에 대해서 각각의 함수를 적용한 값을 리스트로 yield [a, b, c]
def index_(*fn):
    conposed_fn = [make_indexer(f) for f in fn]
    def index__(*xn):
        for x in stream(*xn):
            r = []
            try:
                for indexer in conposed_fn:
                    r.append(indexer(x))
            except Exception as e:
                    print(f'index_ ecception')
                    r = None
                    
            if r is not None:
                yield r
                
    return index__


def index_thru(*fn):
    conposed_fn = [make_indexer(f) for f in fn]
    def index__(*xn):
        for x in stream(*xn):
            acc = x
            for indexer in conposed_fn:
                try:
                    acc = (indexer(x))
                    yield acc
                except Exception as e:
                    print(f'index_thru exception e={e}')
            yield acc
    return index__


def groupby(fn_list, *xn):
    v_stream = stream(*xn)
    indexer = [make_indexer(f) for f in fn_list]
    max_depth = len(indexer) -1
    
    grouped = {}
    indexed = []
    result = {}
    y = 0 # 평면상 y위치
    
    #  groupby 조건이 하나일 경우, 바로 인덱스 
    if max_depth == 0:
        indexer = indexer[0]
        for x in v_stream:
            k = indexer(x)
            # print('groupby k ===>', k)
            if not k in grouped:
                grouped[k] = []
                
            grouped[k].append(x)
            
        result['grouped'] = grouped
        result['indexed'] = indexed
        
        return result        
    
    
    def index_(node, x, y, v, node_indexed, max_depth, indexer):
        k = indexer[x](v)
        print('index_------>k', k)
        node_indexed.append(k)
        
        if x < max_depth:
            # node[k] = node[k] || (x < max_depth - 1 ? {} : []);
            if node[k]:
                node[k] = node[k]
            else:
                node[k] = {} if x < max_depth else []
                
            x = x + 1
            return index_(node[k], x, y, v, node_indexed, max_depth, indexer)
        else:
            node.append(k)

        return node_indexed
    
    g = {}
    
    for data in v_stream:
        node_indexed = []
        node_indexed = index_(grouped, 0, y, data, node_indexed, max_depth, indexer)
        indexed.append(node_indexed)
        
    result = {}
    result['grouped'] = grouped
    result['indexed'] = indexed
    
        
    return result
    

def reduce_(iv, f):
    fn = functor(f)

    def reduce__(*xn):
        acc = iv
        for v in stream(*xn):
            acc = fn(acc, v)
        return acc
    return reduce__


def compose(*fn):
    return [functor(f) for f in fn]


def go(*fn):
    fn_composed = compose(*fn)

    def through_x(x):
        acc = x
        for f in fn_composed:
            acc = f(acc)
        return acc
    return through_x


def rgx_sub(pattern, replace):
    def target(x):
        return re.sub(pattern, replace, x)
    return target


def rand():
    return random.random()


def choice(list):
    return random.choice(list)


def rand_box(list):
    box = list
    def choice_():
        return random.choice(box)

    return choice_

# def rand_choice_in(box, choice_box, cnt):
#     return [x for ]
#     1


def rand_range(start, stop):
    random.randrange(start, stop)


def uniq_rand_range(start, stop, count):
    uniq = {}

    start = start if is_int(start) else 0
    stop = stop if is_int(stop) else 0
    if start == stop:
        return None

    count = count if is_int(count) else 1
    count = count if count >= 0 else count * -1
    if count <= 0:
        return None

    tot_cnt = 0
    try_cnt = 0
    max_cnt = count * 10  # * 10으로 try를 충분히 하도록 하자.

    while (tot_cnt < count) and (try_cnt < max_cnt):
        try_cnt += 1

        x = random.randrange(start, stop)
        if x in uniq:
            continue
        else:
            uniq[x] = 1
            tot_cnt += 1
            yield x

def dict_to_item_stream(dict):
    for k in dict.keys():
        yield {k: dict.get(k)}

def shuffle(*xn):
    list = [x for x in stream(*xn)]
    random.shuffle(list)
    return list

def shuffle_to_stream(*xn):
    return (x for x in shuffle(*xn))

def to_gen(x):
    if is_list(x):
        for item in x:
            yield item
    elif is_dict(x):
        for k, v in x.items():
            yield {k: v}
    elif is_gen(x):
        yield from x
    elif is_iter_but_str(x):
        for _ in x:
            yield _
    else:
        yield x


#기본 제공하는 json보다 성능상 이점 있음
def json_to_dict(json_data):
    r = None
    try:
        r = orjson.loads(json_data)
    except:
        r = None
    return r


def json_file_to_dict(file, encoding = 'utf-8'):
    dict = None
    try:
        with open(file, 'r', encoding=encoding) as f:
            json_data = f.read()
            dict = orjson.loads(json_data)
    except BaseException as err:
        print(err)

    return dict

def file_to_dict_stream(file):
    for line in line_stream(file):
        try:
            json = orjson.loads(line)
            yield json
        except Exception as e:
            print(f'file_to_dict_stream exception err={e}')
            continue

def dict_stream_from_line(line_stream):
    for line in stream(line_stream):
        try:
            o = orjson.loads(line)
            yield o
        except Exception as e:
            print(f'dict_stream_from_line error={e}, line={line}')
            continue

def dict_to_list(dict):
    return [{k: v} for k, v in dict.items()]

def mmap_line_stream(path, pattern=None): 
    with open(path, "rb", buffering=0) as f:
        m = mmap.mmap(f.fileno(), 0,access=mmap.ACCESS_READ)
        while True:
            line = m.readline()
            if line:
                yield line.decode("utf-8").rstrip()
                # lines.append(line.decode("utf-8"))
            else:
                break
        m.close()


def text_stream_from_file(file_stream):
        """
        전체파일을 읽는다
        """
        # file_path_name이 None 아니면 새로 세팅 아니면 그대로 이용 
        
        for file_path_name in stream(file_stream):
            if os.path.exists(file_path_name) and os.path.isfile(file_path_name):
                try:
                    with open(file_path_name, 'r', encoding='utf-8') as fp:
                        text = ''.join( fp.readlines() )
                        yield text
                    
                except OSError:
                    print(f'file open error : file : {file_path_name}, err={OSError}')
                
            else:
                print(f'file open error : file : {file_path_name}')
                
                
            
####################### Regex 패턴관련  #######################
# pattern 합친다
def join_patten(*patterns): return '.*?'.join([p for p in patterns])

# 클로져 함수 : 패턴을 가변인자로 받아 여러개에 모두 해댱해야 True를 리턴
def has_pattern(*patterns):
    ptc = re.compile(join_patten(*patterns))
    def search(v):
        return False if ptc.search(v) is None else"true"
    return search
# print('has pattern -------------->', has_pattern('a', 'b')('a.................b'))


# 클로져 함수 : 패턴을 가변인자로 받아 하나라도 해당하면 True 리턴
def has_pattern_in(*patterns):
    pattern_list = [re.compile(p) for p in patterns]
    def search(v): 
        for pattern in pattern_list:
            if pattern.search(v) is None:
                return False
        return"true"
    return search

# api_url = re.sub("\d{1,}", '{no}', api_uri)
def pattern_replace_x(pattern, replaced, x):
    return re.sub(pattern, replaced, x)

def replace(matched_pattern, repalce_str, src):
    return re.sub(matched_pattern, repalce_str, src)

def replace_that(source, replaced, *pattern):
    for p in pattern:
        source = re.sub(p, replaced, source)
        
    return source

def search_pattern(name, flags=re.DOTALL, *patterns):
    joined_pattern = join_patten(*patterns)
    ptc = re.compile(joined_pattern, flags)
    def search(v):
        r = ptc.search(v)
        if r:
            return {
                'name' : name, 
                'match' :v[r.start():r.end()],
                'group' :r.groups(),
                'groups' :r.groupdict(),
            }
        return None
    return search

def search_pattern_group(name, *patterns):
    joined_pattern = join_patten(*patterns)
    ptc = re.compile(joined_pattern)
    def search(v):
        search_ret = ptc.search(v)
        if search_ret is not None:
            return search_ret.groupdict()[name]
        return None
    return search

def rgx_capture(name, *pattern):
    return search_pattern_group(name, *pattern)


def path_only(filepath):
    return os_nix_splitter.join(
        filepath.replace(os_win_splitter, os_nix_splitter).split(os_nix_splitter)[:-1]
    ) + os_nix_splitter
    
# path를 제외한 파일명만 얻는다    
def file_only(filepath):
    return filepath.replace(os_win_splitter, os_nix_splitter).split(os_nix_splitter)[:-1]

def extension_only(filename):
    file_splitted = filename.split('.')
    return file_splitted[-1] if len(file_splitted) > 1 else ''
    
def file_only_but_extension(file_path_name):
    file_splitted = file_path_name(os_win_splitter, os_nix_splitter).split(os_nix_splitter)[:-1]
    file_name = file_splitted[0:-1] if len(file_splitted) > 1 else file_splitted
    return ''.join(file_name)

def make_file_name(path, file):
    path = path.replace(os_win_splitter, os_nix_splitter)
    file_made = path + file if path[-1] == '/' else path + os_nix_splitter + file
    return file_made

def make_dir(*path_list):
    dir_path = "/".join(path_list)
    if os.path.exists(dir_path):
        # print('dir_path already', dir_path)
        1
    else:
        os.mkdir(dir_path)
        
    return dir_path

def rpad(src, pad_count, pad_char):
    return str(src).rjust(pad_count, pad_char)

def lpad(src, pad_count, pad_char):
    return str(src).ljust(pad_count, pad_char)

def comma(num):
    if is_num(num):
        return format(num, ',')
    else:
        return str(num)
    
def com(num):
    if is_num(num):
        return format(num, ',')
    else:
        return str(num)
    
    
def split_file(one_file, target_path=None, split_count=5, idx_ch="_", remove_src=False):
    
    if is_file(one_file) and is_int(split_count) and split_count > 1:
        fp_list = []
        
        # print(f'split count={split_count}')
        
        path = os.path.abspath(one_file)
        # print('-------------------------->', path)
        
        path, file, ext = file_path_name_ext(one_file)
        for fidx in range(0, split_count):
            fn_splitted = path_concat(target_path, file + f'{idx_ch}{fidx}' + f'.{ext}')
            # print('split file ->', fn_splitted)
            
            fp = open(fn_splitted, 'w', encoding='utf-8')
            fp_list.append(fp) if fp else 1
            
        
        fp_cnt = len(fp_list)
        
        line_idx = 0
        for line in line_stream(one_file):
            fp_idx = line_idx % fp_cnt
            fpw = fp_list[fp_idx]
            fpw.write(line)
            line_idx += 1
            
            if line_idx % 100_0000 == 0:
                print('line_idx ->', comma(line_idx))
                
        [fp.close() for fp in fp_list if fp]
        
        if remove_src:
            rm_file(one_file)
            
        print('tot_line_count : ', comma(line_idx))    
        return"true"
        
    else:
        print(f'split_file : param ={one_file}, {target_path}, {split_count}, {idx_ch}, {remove_src}')
        return None
    
        

class xlogger:
    def __init__(self, file, append = False, std_out = True):
        #append 방식으로 파일을 연다
        file_open_mode = 'a' if append else 'w'
        self.std_out = std_out
        
        try:
            self.fp = open(file, file_open_mode, encoding='utf-8')
        except Exception as e:
            self.fp = None
            
    def log(self, v):
        self.fp.write(v + '\n') if self.fp else 1
        print(v) if self.std_out else 1
        
    
    def close(self):
        if self.fp:
            self.fp.close()
    
            
class xFileWriter:
    def __init__(self, path_file_name, encode='utf-8'):
        try:
            self.path_file_name = path_file_name
            self.fp = open(path_file_name, 'w', encoding = encode)
        except OSError:
            print(f'file open error : file={path_file_name}, err={OSError}')
            self.fp = None
    
    def write_line(self, v, new_line_append=True):
        if self.fp:
            try:
                if new_line_append:
                    self.fp.write(v + '\n')
                else:
                    self.fp.write(v)
            except Exception as e:
                print(f'write line error={e}')
                
    def write_line_from(self, *xn):
        for x in stream(*xn):
            self.write_line(x)
    
    def close(self):
        if self.fp:
            self.fp.close()
    
    def get_path_file(self):
        return self.path_file_name
    
####################### file ops #######################
def file_path_stream(tar_get_dir, file_pattern=None):
    ptc_file_pattern = re.compile('.*') if is_in(file_pattern, None, '', '*', False) else re.compile(file_pattern) 
    
    for cur_dir_path, sub_dir_list, cur_dir_file_list in os.walk(tar_get_dir):
        for name in cur_dir_file_list:
            file_path_name = os.path.join(cur_dir_path, name).replace('\\', '/')
            if ptc_file_pattern.search(file_path_name):
                yield file_path_name
                
def file_path_list(root_dir, file_pattern=None):
    return [file for file in file_path_stream(root_dir, file_pattern)]


def dir_path_stream(target_dir_path, dir_pattern=None):
    ptc_file_pattern = re.compile('.*') if is_in(dir_pattern, None, '', '*') else re.compile(dir_pattern) 
    
    for cur_dir_path, sub_dir_list, cur_dir_file_list in os.walk(target_dir_path):
        for sub_dir in sub_dir_list:
            dir_path = os.path.join(cur_dir_path, sub_dir).replace('\\', '/')
            if ptc_file_pattern.search(dir_path):
                yield dir_path
            
def dir_path_list(root_dir, dir_pattern=None):
    return [file for file in dir_path_stream(root_dir, dir_pattern)]


def line_generator(file_path_name):
    if os.path.isfile(file_path_name):
        try:
            fp = open(file_path_name, "r", encoding='utf-8')
        except:
            print('file open error : ', fp)
            fp = None
            return None
    else:
        print("file nothing=>", file_path_name)
        return None
    
    for line in fp:
        yield line

def str_to_bytes(v, unit='gb'):
    v = replace(r'[\s]{1,}', '', v).lower()
    ptc = re.compile(r'^(?P<num>.*?)(?P<unit>[a-zA-Z]{0,})$')
    search_ret = ptc.search(v)
    if search_ret:
        num = float(search_ret.groupdict()['num'])
        unit = search_ret.groupdict()['unit']
        r = 0.0
        if unit == 'kb':
            r = num * 1024
        elif unit == 'mb':
            r = num * 1024 * 1024
        elif unit == 'gb':
            r = num * 1024 * 1024 * 1024
        elif unit == 'tb':
            r = num * 1024 * 1024 * 1024 * 1024
        elif unit == 'pb':
            r = num * 1024 * 1024 * 1024 * 1024 * 1024
        
    else:
        r = 0

    return r

# path file 관련 
def line_stream(file_stream, new_line_remove = True):
    for file in stream(file_stream):
        with open(file, 'r', encoding='utf-8') as fdata:
            if new_line_remove:
                for line in fdata:
                    try:
                        line = re.sub(r'[\r\n]$', '', line)
                        yield line
                    except Exception as e:
                        print(f'line_stream error e={e}, file={file}, new_line_remove=True')
                        continue
            else:
                for line in fdata:
                    yield line
                    
def line_stream_with_cnt(file, new_line_remove=True):
    cnt = 0
    with open(file, 'r', encoding='utf-8') as fp:
        cnt += 1
        if new_line_remove:
            for line in fp:
                yield re.sub(r'[\r\n]$', '', line), cnt
        else:
            for line in fp:
                yield line, cnt
    

def file_to_line_stream(file_path_name, new_line_remove = True):
    with open(file_path_name, 'r', encoding='utf-8') as fdata:
        if new_line_remove:
            for line in fdata:
                yield line.strip()
        else:
            for line in fdata:
                yield line

def filestream_to_line_stream(file_stream, new_line_remove = True):
    for file in stream(file_stream):
        with open(file, 'r', encoding='utf-8') as fdata:
            if new_line_remove:
                for line in fdata:
                    # yield re.sub(r'[\r\n]$', '', line)
                    yield line.strip()
                    
            else:
                for line in fdata:
                    yield line
                    
def concat_files(out_file, start_path, file_pattern='*'):
    file_stream = file_path_stream(start_path, file_pattern)
    with open(out_file, 'wb') as outfile:
        for file in file_stream:
            if file == out_file:
                continue
            print('concat_files target : ', file)
            with open(file, 'rb') as readfile:
                shutil.copyfileobj(readfile, outfile)    
                
                
def extract_tar(tar_path, target_path, mode="r:gz", target_path_force_make=True):
    # target path 필요시 선생성
    if target_path_force_make:
        if is_dir(target_path) is False:
            mkdir(target_path)
            
    else:
        if is_dir(target_path) is False:
            print(f'extract_tar except target_path={target_path} not exist')
            return False
            
    try:
        with tarfile.open(tar_path, mode) as tar:
            file_names = tar.getnames()
            for file_name in file_names:
                tar.extract(file_name, target_path)
        return"true"

    except Exception as e:
        print(tar_path, 'extranct error ====>', e)
        return None
    
def remove_file(file_path_name):
    try:
        os.remove(file_path_name)
        return"true"
    except Exception as e:
        print(file_path_name, 'remove_file error ====>', e)
        return False
    
def remove_files(*xn):
    for file in stream(*xn):
        strfile = str(file)
        if os.path.isfile(strfile):
            try:
                os.remove(strfile)
            except:
                print('file remove got exception :', strfile)
        else:
            print('file does not exist :', strfile)   


def is_exist(path_or_file_name):
    try:
        is_file_exist = os.path.exists(path_or_file_name)
        return is_file_exist
    except Exception as e:
        print(f'is_exist except={e}, path_or_file_name={path_or_file_name}')
        return False
    
    
def is_dir(dir_path):
    if is_exist(dir_path):
        try:
            return True if os.path.isdir(dir_path) else False
        except Exception as e:
            print(f'is_dir except e={e}, dir_path={dir_path}')
            return False
    else:
        return False

def is_file(file_path):
    if is_exist(file_path):
        try:
            return True if os.path.isfile(file_path) else False
        except Exception as e:
            print(f'is_dir except e={e}, is_file={file_path}')
            return"true"
    else:
        return False
    
def file_path_name_ext(path_file_name):
    path = ''
    file = ''
    extension = ''
    if path_file_name is None or is_str(path_file_name) is None or len(path_file_name) <= 0:
        print('file format error path_file_name={path_file_name}')
    else:
        splitted_path_file_name = path_file_name.replace(os_win_splitter, os_nix_splitter).split(os_nix_splitter)
        len_splitted_path = len(splitted_path_file_name)
        if splitted_path_file_name == 1:
            file = path_file_name
        elif len_splitted_path > 1: 
            path = '/'.join(splitted_path_file_name[:-1]) + '/'
            file_name_ext = splitted_path_file_name[-1]
            file_name_ext_splitted = file_name_ext.split('.')
            if len(file_name_ext_splitted) > 1:
                file = '.'.join(file_name_ext_splitted[:-1])
                extension = file_name_ext_splitted[-1]
            else:
                file = file_name_ext_splitted[0]
                
    return (path, file, extension)
            

def rm_dir(target_path):
    if is_dir(target_path):
        try:
            shutil.rmtree(target_path)
            return"true"
        except Exception as e:
            print('rm_dir exception : ', e)
            return"true"
    else:
        print('path is not exist or is not dir, target_path :', target_path)
        return False
    
# in은 subdirectory를 지움을 나타냄    
def rm_sub_dir(parent_path, pattern):
    #본인 path는 제외
    sub_dir_path_list = [x for x in dir_path_list(parent_path, pattern) if x != parent_path]
    for dir_path in sub_dir_path_list:
        rm_dir(dir_path)

def rm_file(file_path_name):
    #파일이 존재하지 않을 때
    if is_exist(file_path_name) is False:
        print(f'rm_file exception, file_path is not exist, file={file_path_name}')
        return False
    
    #파일이 존재할 때 
    if is_file(file_path_name):
        try:
            os.remove(file_path_name)
            return"true"
        except Exception as e:
            print('rm_file exception', e, file_path_name)
            return False
    else:
        print(f'rm_file exception,  file={file_path_name} is not exsit ')
        return False
        
def rm_files(files, pattern=None):
    rm_file_pattern = re.compile('.*') if is_in(pattern, None, '', '*') else re.compile(pattern)
    for file in stream(files):
        if rm_file_pattern.search(file):
            return rm_file(file)
        
        if os.path.isfile(file) and rm_file_pattern.search(rm_file):
            try:
                os.remove(rm_file)
            except:
                print('file remove got exception :', file)
        else:
            print('file does not exist :', file) 
        
def mkdir(target_path, over_write = False):
    #over_write 이면서 기존에 디렉토리가 존재하면 기존것을 지우고 새로 생성
    if is_dir(target_path):
        if over_write:
            rm_dir(target_path)
        else:
            return target_path
    else:
        try:
            os.makedirs(target_path)   
            return target_path 
        except Exception as e:
            print(f'mkdir({target_path}) exception={e}')
            return None
        
def mkdirs(*path):
    for p in path:
        mkdir(p)

def rm_file_in(parent_path, pattern = False):
    if is_file(parent_path):
        rm_file(parent_path)
    elif is_dir(parent_path):
        target_file_list = file_path_list(parent_path, pattern)
        rm_files(target_file_list)
        
            
            
def file_size(path, size_char = None):
    ch = '' if size_char is None else size_char.lower()[0]
    devider = 1
    if ch == 'k':
        devider = 1024
    elif ch == 'm':
        devider = 1024 * 1024
    elif ch == 'g':
        devider = 1024 * 1024 * 1024
    elif ch is None:
        devider = 1
    else:
        devider = 1
        
    if is_file(path) or is_dir(path):
        size = os.path.getsize(path)
        
        if devider == 1:
            return size
        else:
            return size, round(size / devider, 3)
    else:
        return None
        
        
class xcsv:
    def __init__(self, file_path_name):
        self.file = file_path_name
        
    def replace(self, pattern, replaced, x):
        ptc = None
        try:
            ptc = re.compile(pattern)
        except:
            print('pattern is not correct')
            return None
        
        return re.sub(ptc, replaced, x)
    
    def to_csv_stream(self, file_path_name=None, splitter = ',', newline_remove=True):
        file = file_path_name if file_path_name else self.file
        ptc = re.compile(r'\s{1,}')
        with open(file, 'r', encoding='utf-8') as fp:
            for line in fp:
                line_csv = self.replace(ptc, ',', line)
                if newline_remove:
                    line_csv = self.replace(r'[\r\n],', '', line_csv) 
                yield line_csv
                
    def to_csv_file(self, file_path_name, csv_file):
        file = file_path_name if file_path_name else self.file
        csv_stream = self.to_csv_stream(file)
        
        line_stream_to_file(csv_file, csv_stream)
        
    def stream_to_csv_file(csv_file, header, csv_stream):
        1
        
class xFilePath:
    def __init__(self, root_dir=None, file_pattern=None, reject_pattern = None):
        self.root_dir = root_dir
        self.file_pattern = file_pattern
        self.reject_pattern = reject_pattern
        self.file_list = []
        pass
    
    def file_path_stream(self, root_dir=None, file_pattern=None, reject_pattern = None):
        self.root_dir = root_dir if root_dir else self.root_dir
        self.file_pattern = file_pattern if file_pattern else self.file_pattern
        self.reject_pattern = reject_pattern if reject_pattern else self.reject_pattern
        
        ptc_file_pattern = re.compile(file_pattern)
        
        for root, dirs, files in os.walk(root_dir):
            for name in files:
                file_path_name = os.path.join(root, name).replace(_PATH_SPLITTER_, '/')
                if ptc_file_pattern.search(file_path_name):
                    yield file_path_name
                    
    def file_path_list(root_dir, file_pattern):
        return [file for file in file_path_stream(root_dir, file_pattern)]
                    
# 하나의 텍스트파일을 담당
class xTextFileReader:
    def __init__(self, file_path_name):
        """_summary_
        생성자에서 file_path를 주면 명시적으로 open할 필요없다
        Args:
            file_path_name (_type_): _description_
        """
        self.file_path_name = file_path_name
        self.fp = None
        
        # check
        if os.path.exists(self.file_path_name) and os.path.isfile(self.file_path_name):
            try:
                self.fp = open(file_path_name, 'r', encoding='utf-8')
            except OSError:
                self.fp = None
                print(f'file open error : {file_path_name}, err={OSError}')
            
        else:
            print(f'file open error : {file_path_name}')
        
    def open(self, file_path_name = None):
        """_summary_
        생성자외에 명시적으로 파일을 오픈한다
        기존 생성자에서 파일을 열었다고 하더라고 기존것은 닫고 새로 file pointer(fp)를 연다
        Args:
            file_path_name (_type_): _description_
        """
        if file_path_name is None:
            print('file_name is None check')
            return self
        
        self.file_path_name = file_path_name
        self.fp = None
        
        # check
        if os.path.exists(self.file_path_name) and os.path.isfile(self.file_path_name):
            try:
                self.fp = open(file_path_name, 'r', encoding='utf-8')
            except OSError:
                print(f'file open error : file={file_path_name}, err={OSError}')
            
        else:
            print(f'file open error : file{file_path_name}')
    
    def read(self, file_path_name=None, close=False):
        """
        전체파일을 읽는다
        """
        # file_path_name이 None 아니면 새로 세팅 아니면 그대로 이용 
        
        if file_path_name:
            open(file_path_name)
            
        if self.fp:
            with self.fp:
                text = ''.join( self.fp.readlines() )
                if close : self.fp.close()
                return text
        else:
            print('self.fp is None')
            
            
    def line_stream(self):
        if self.fp is not None:
            for line in self.fp:
                yield line
                
    def lines(self):
        if self.fp:
            lines = self.fp.readlines()
            return lines
        
    def close(self):
        if self.fp: elf.fp.close()
            
def mmap_line_count(filename):
    f = open(filename, "r+")
    buf = mmap.mmap(f.fileno(), 0)
    lines = 0
    readline = buf.readline
    while readline():
        lines += 1
    return lines            
                      
def line_generator(file_path_name):
    if os.path.isfile(file_path_name):
        try:
            fp = open(file_path_name, "r", encoding='utf-8')
        except:
            print('file open error : ', fp)
            fp = None
            return None
    else:
        print("file nothing=>", file_path_name)
        return None
    
    for line in fp:
        yield line
                
def line_stream_from_file(file_stream, remove_new_line = True):
    yield from line_stream(file_stream, remove_new_line)
    
def line_stream_to_file(file_with_path, line_stream, newline_append = True):
    with open(file_with_path, 'w', encoding='utf-8') as f:
        print('line_stream_to_file ->', file_with_path)
        line_cnt = 0
        if newline_append:
            for line in stream(line_stream):
                line_cnt = line_cnt + 1
                if line_cnt % 10000 == 0:
                    print(line_cnt)
                f.write(line + '\n')
        else:
            line_cnt = 0
            for line in stream(line_stream):
                line_cnt = line_cnt + 1
                if line_cnt % 10000 == 0:
                    print(line_cnt)
                f.write(line)
                
def flush_stream(xstream, debug = None, interval = 100):
    if debug:
        interval = interval if is_int(interval) else 100
        interval = interval if interval > 0 else interval * -1
        idx = 0
        debug_fn = functor(debug)
        for x in stream(xstream):
            idx += 1
            if idx % interval == 0:
                debug_fn(x, idx)

    for x in stream(xstream):
        x           

def log_stream(xstream, interval = 100):
    print(xstream, interval)
    interval = interval if is_int(interval) else 100
    interval = interval if interval > 0 else interval * -1
    idx = 0
    for x in stream(xstream):
        idx += 1
        if idx % interval == 0:
            print(idx, x)
        yield x
        
def get_uniq_stream(stream, fn):
    uniq_dict = {}
    
    
class xConfiguration:
    def __init__(self, profile='dev', config={}):
        self.profile = profile
        self.config = {}
        pass  
    
    def set_config(self, config = {}):
        if config is not None and type(config) is dict:
            self.config = config
            
        return self
    
    def set(self, key, value):
        if key is None or key == '':
            return self
        
        self.config[key] = value
        return self
    
    def get(self, key):
        return self.config[key]
        
class xIndexer:
    def __init__(self):
        self.indexer_list = []
        self.stream_ = None
    
    def set(self, *fn):
        for f in fn:
            self.indexer_list.append(make_indexer(f))
        return self
    
    def add_indexer(self, f):
        self.indexer_list.append(make_indexer(f))
        return self
    
    def set_indexer(self, *fn):
        self.indexer_list.clear()
        for f in fn:
            self.add(f)
        return self
        
    def index(self, x):
        return [f(x) for f in self.indexer_list]
                
    def index_at(self, order, x):
        return self.indexer_list[order](x)
    
    def index_through(self, x):
        acc = x
        for indexer in self.indexer_list:
            acc = indexer(acc)
        return acc
    
    def from_stream(self,*xn):
        self.stream_ = stream(*xn)
        return self
        
    def to_stream(self, at=None, ):
        for x in self.stream_:
            indexeded = self.index(x)
            yield indexeded
            
    def to_stream_at(self, order):
        for x in self.stream_:
            indexeded = self.index_at(order, x)
            yield indexeded
            
    def to_stream_througth(self):
        for x in self.stream_:
            indexeded = self.index_through(x)
            yield indexeded
                
                
    
class xUnique:
    def __init__(self):
        # self.indexer = xIndexer()
        self.stream_  = None
        self.uniq_list = []
        self.fn_list = []
        self.name_list = []
        self.tot_cnt_list = []
        
        
    def add_uniq(self, name, f):
        self.name_list.append(name)
        self.uniq_list.append({})
        self.fn_list.append(make_indexer(f))
        return self
        
        
    def from_stream(self, *xn):
        self.stream_ = stream(*xn)
        return self
    
    def from_list(self, list):
        idx_list = [i for i in range(0, len(self.fn_list))]
        for x in stream(list):
            for idx in idx_list:
                fn = self.fn_list[idx]
                uniq = self.uniq_list[idx]
                v = fn(x)
                uniq[v] = uniq[v] + 1 if v in uniq else 1
        return self
    
    def to_stream(self):
        idx_list = [i for i in range(0, len(self.fn_list))]
        for x in self.stream_:
            for idx in idx_list:
                fn = self.fn_list[idx]
                uniq = self.uniq_list[idx]
                v = fn(x)
                uniq[v] = uniq[v] + 1 if v in uniq else 1
            yield x
        
    def flush(self, debug=None):
        i = 0
        
        if debug is None:
            for x in self.to_stream():
                x
        else:
            for x in self.to_stream():
                x if debug is None else functor(debug)(x, i)
                i += 1
        return self
            
    def uniq_of(self, index=None):
        if is_int(index) and index >=0 and index < len(self.uniq_list):
            return self.uniq_list[index]
        elif index == None:
            return self.uniq_list
        
        for i, n in enumerate(self.name_list):
            if n == index:
                return self.uniq_list[i]
            
        return None
    
    def list_of(self, index=None):
        if is_int(index) and index >=0 and index < len(self.uniq_list):
            return [x for x in self.uniq_list[index].keys()]
        
        elif index == None:
            return self.uniq_list
        
        for i, n in enumerate(self.name_list):
            if n == index:
                return [x for x in self.uniq_list[i].keys()]
            
        return None
    
    def count_of(self, index):
        if is_int(index) and index >=0 and index < len(self.uniq_list):
            return [v for v in self.uniq_list[index].values()]
        
        for i, n in enumerate(self.name_list):
            if n == index:
                return [v for v in self.uniq_list[i].values()]
            
    def tot_count_of(self, index):
        if is_int(index) and index >=0 and index < len(self.uniq_list):
            return sum( [v for v in self.uniq_list[index].values()])
        
        for i, n in enumerate(self.name_list):
            if n == index:
                return sum( [v for v in self.uniq_list[i].values()])
            
    
class xStream:
    def __init__(self) -> None:
        self.indexer = xIndexer()
        self.pipe_line = []
        self.stream_ = None
        
    def instance(self, *args):
        return self
    
    def from_stream(self, *xn):
        self.stream_ = stream(*xn)
        return self
    
    def to_stream(self, peeker=None):
        acc_stream = self.stream_
        for pipe in self.pipe_line:
            acc_stream = pipe(acc_stream)
            
        if peeker is None:
            for x in acc_stream:
                yield x    
        
        else:
            pf = functor(peeker)
            idx = 0
            for x in acc_stream:
                pf(x, idx)
                idx += 1
                yield x
        
    def map(self, *fn):
        for f in fn:
            self.pipe_line.append(map_(f))
        return self
    
    def filter(self, *fn):
        for f in fn:
            self.pipe_line.append(filter_(f))
        return self
    
    def index(self, *fn):
        self.pipe_line.append(index_(*fn))
        return self
    
    def index_thru(self, *fn):
        self.pipe_line.append(index_thru(*fn))
        return self
        
    def collect(self, f=None):
        mapper = f if is_fun(f) else identity
        return [mapper(acc) for acc in self.to_stream()]
    
    # def unique(self, *fn):
            
def max(a, b):
    return b if b > a else a

def min(a, b):
    return b if b > a else a

def group_by_cnt(cnt, input):
    return [input[i : i+cnt] for i in range(0, len(input), cnt)]  
        
class xGrouper:
    def __init__(self):
        self.indexer = xIndexer()
        self.gname = []
        self.grouped = {}
        
        self.agg = []
    
    def group_by(self, grouper, group_name):
        if is_str(group_name) == False or len(group_name) <= 0:
            group_name = 'g_' + str(len(self.group_name))
        
        self.indexer.add(grouper)
        agg = {
            'tot_cnt' : 0,
            'total' : 0,
            'mean'  : 0,
            'uniq'  : 0,
            'max'   : 0,
            'min'   : 0,
        }
        self.agg.append(agg)
        self.gname.append(group_name)
        return self
    
    # 각각의 groupby에 대해서 grouping을 한다
    def group(self, x):
        indexed_list = self.indexer.index(x)
        
        idx = 0
        # for indexed in indexed_list:
            
            # agg = self.agg[idx]
            # agg['count'] = agg['count'] + 1
            
        # key = str(idx_val)
        # if idx_val in self.grouped:
        #     self.agg[key]['count'] = self.agg[key]['count'] + 1
        # else:
        #     self.grouped[key] = []
        #     self.grouped[key].append(x)
            
    # 최종 indexd 된 결과를         
    # def group_through(self, x, f):
        
            
            
    # def get_count(self):
        
            
            
    # def grouped(self):
    #     return self.grouped
            
    
# 스트림을 받아(stream) 신규 스트림(버퍼시작) 조건을 만나기 전까지 버퍼에 저장하고 신규스트림 생성조건을 만나면 기존 버퍼에 저장되어 있는 것을 합쳐서(reduce) 신규스트림 시작으로 yield 한다
class GroupStream:
    def __init__(self): 
        self.buf = []
        self.groupers = []
        self.bufidx = 0
        self.reducer = None

    def add_grouper(self, grouper): 
        self.groupers.append(grouper)
        return self

    def set_reducer(self, reducer): 
        self.reducer = reducer
        return self

    def stream(self, stream):
        for v in stream:
            if self.detect(v) and len(self.buf) > 0:    # 신규스트림(버퍼링) 조건이고 버퍼에 내용이 있다면 
                yield self.reducer(self.buf)            #버퍼에 내용물을 합쳐서 yield 하고 
                self.buf.clear()                        #내보낸 다음 버퍼는 비우고
            self.buf.append(v)                          #버퍼에 추가
            
        # 신규 그룹이 나타나지 않더라도 그룹에 남은 것이 있으면 모아서 다시 흘려 보낸다.
        if len(self.buf) > 0:
            yield self.reducer(self.buf)
            
        self.buf.clear()

    def detect(self, v):
        for grouper in self.groupers:
            if grouper(v):
                return"true"
        return False

def buf_stream(dectector, reducer):
    buf = []
    reduce_fn = functor(reducer) if reducer else identity
    def stream(stream):
        for v in stream:
            if dectector(v) and len(buf) > 0:    # 신규스트림(버퍼링) 조건이고 버퍼에 내용이 있다면 
                yield reduce_fn(buf)            #버퍼에 내용물을 합쳐서 yield 하고 
                buf.clear()                        #내보낸 다음 버퍼는 비우고
            buf.append(v)                          #버퍼에 추가
            
        if len(buf) > 0:
            yield reduce_fn(buf)
        buf.clear()
    return stream

def group_stream(dectector, reducer=None):
    buf = []
    reduce_fn = functor(reducer) if reducer else identity
    group_start_finder = predicator(dectector)
    
    def group_by(*xn):
        for v in stream(*xn):
            if group_start_finder(v) and len(buf) > 0:    # 신규스트림(버퍼링) 조건이고 버퍼에 내용이 있다면 
                yield reduce_fn(buf)            #버퍼에 내용물을 합쳐서 yield 하고 
                buf.clear()                        #내보낸 다음 버퍼는 비우고
            buf.append(v)                          #버퍼에 추가
            
        if len(buf) > 0:
            yield reduce_fn(buf)
        buf.clear()
    return group_by

def stream_to_group_stream(dectector, reducer=None):
    buf = []
    reduce_fn = functor(reducer) if reducer else identity
    detector_fn = predicator(dectector)
    def group_by(*xn):
        for v in stream(*xn):
            if detector_fn(v) and len(buf) > 0:    # 신규스트림(버퍼링) 조건이고 버퍼에 내용이 있다면 
                yield reduce_fn(buf)            #버퍼에 내용물을 합쳐서 yield 하고 
                buf.clear()                        #내보낸 다음 버퍼는 비우고
            buf.append(v)                          #버퍼에 추가
            
        if len(buf) > 0:
            yield reduce_fn(buf)
        buf.clear()
    return group_by

def stream_to_group_list(dectector, reducer=None):
    buf = []
    reduce_fn = functor(reducer) if reducer else identity
    detector_fn = predicator(dectector)
    def group_by(*xn):
        for v in stream(*xn):
            if dectector(v) and len(buf) > 0:    # 신규스트림(버퍼링) 조건이고 버퍼에 내용이 있다면 
                yield reduce_fn(buf)            #버퍼에 내용물을 합쳐서 yield 하고 
                buf.clear()                        #내보낸 다음 버퍼는 비우고
            buf.append(v)                          #버퍼에 추가
            
        if len(buf) > 0:
            yield reduce_fn(buf)
        buf.clear()
    return group_by

def vectorize_fn(f):
    return 

# value는 반드시 숫자형(int) 이어야 한다
def item_cnt(item_count_dict, dict):
    for x, c in dict.items():
        item_count_dict[x] = item_count_dict[x] + c if x in item_count_dict else c
        
    return item_count_dict

def sum_count_per_item(item_count_dict, dict):
    for x, c in dict.items():
        item_count_dict[x] = item_count_dict[x] + c if x in item_count_dict else c
        
    return item_count_dict

def uniq_from(*xn):
    uniq = {}
    for x in stream(*xn):
        uniq[x] = uniq[x] + 1 if x in uniq else 1
        
    return uniq
        
def sort_by_value(dict, reverse_yn=True):
    sorted_list = sorted(dict.items(), key = lambda item: item[1], reverse = reverse_yn)
    result_dict = {}
    
    for kv in sorted_list:
        result_dict[kv[0]] = kv[1]
    
    return result_dict
    
def sort_by_key(dict, reverse_yn=True):
    sorted_list = sorted(dict.items(), key = lambda item: item[0], reverse = reverse_yn)
    result_dict = {}
    
    for kv in sorted_list:
        result_dict[kv[0]] = kv[1]
    
    return result_dict


def gzip_str(string_: str) -> bytes:
    return gzip.compress(string_.encode())

def path_concat(*path):
    path_list = [p.replace(os_win_splitter, os_nix_splitter)  for p in path]
    path_list = [p[:-1] if p[-1] == '/' else p for p in path_list]
    path_list = [str(p) for p in path_list]
    path = '/'.join(path_list)
    
    return path

def path_file_ext(path, file, ext):
    path = path.replace(os_win_splitter, os_nix_splitter)
    path = path if path[-1] == '/' else path + '/'
    ext = ext if ext[0] == '.' else '.' + ext
    return path + file + ext

def zip_argulist(arg_list, *params):
    a_list =[ x for x in stream(arg_list)]
    p_list = [p for p in stream(*params)]
    r = []
    
    arg_idx = 0
    for a in a_list:
        temp = []
        
        temp.append(arg_idx) #argument index,
        temp.append(a)       
        
        for p in p_list:
            temp.append(p)
            
        r.append(temp)
        arg_idx += 1
        
    return r
         

#자릿수 판별
def digit_len(n):
    len = 1
    while True: 
        devided = (n * 1.0) / pow(10, len)
        if (devided) < 1:
            break
        len += 1
    
    return len

def days_in_month(yyyy, mm):
    y = int(yyyy)
    m = int(mm)
    return calendar.monthrange(y, m)[1]
    

def tar_compress(root_dir, file_pattern, tar_path_file):
    if tar_path_file is None or len(tar_path_file) <= 0:
        print(f'path_file is not valied path_file={tar_path_file}')
        return False
    
    file_list = file_path_list(root_dir, file_pattern)
    
    compress = 'w:gz' if tar_path_file.find('tar.gz') >= 1 or tar_path_file.find('.tgz') >= 1 else 'w'
    print('compress ---------------------------->', compress)
    
    with tarfile.open(tar_path_file, compress) as tar:
        for file in file_list:
            tar.add(file)
            print('tar add ---------->', file)
            
    tar.close()
    
def zip_compress(root_dir, file_pattern, zip_path_file):
    if zip_path_file is None or len(zip_path_file) <= 0:
        print(f'path_file is not valied path_file={zip_path_file}')
        return False
    
    file_list = file_path_list(root_dir, file_pattern)
    
    # compress = 'w:gz' if zip_path_file.find('tar.gz') >= 1 or zip_path_file.find('.tgz') >= 1 else 'w'
    
    with ZipFile(zip_path_file, mode='w') as zf:
        for f in file_list:
            zf.write(f)
            
    # zf.close()
    print('zip_compress ---------------------------->', zip_path_file)
    

#디렉토리를 전체 압축한다
def make_tarfile(output_filename, source_dir):
    with tarfile.open(output_filename, "w:gz") as tar:
        tar.add(source_dir, arcname=os.path.basename(source_dir))
        
def tar_comp_dir(output_filename, source_dir):
    try:
        with tarfile.open(output_filename, "w:gz") as tar:
            tar.add(source_dir, arcname=os.path.basename(source_dir))
        return"true"
    except Exception:       
        print(f"tar_comp_dir error ={Exception}")
        return False   
        
        
        
# from datetime import datetime
# now = datetime.now()
# print("현재 : ", now)
# print("현재 날짜 : ", now.date())
# print("현재 시간 : ", now.time())
# print("timestamp : ", now.timestamp())
# print("년 : ", now.year)
# print("월 : ", now.month)
# print("일 : ", now.day)
# print("시 : ", now.hour)
# print("분 : ", now.minute)
# print("초 : ", now.second)
# print("마이크로초 : ", now.microsecond)
# print("요일 : ", now.weekday())
# print("문자열 변환 : ", now.strftime('%Y-%m-%d %H:%M:%S'))
# Output
# 현재 :  2021-12-22 15:46:26.695840
# 현재 날짜 :  2021-12-22
# 현재 시간 :  15:46:26.695840
# timestamp :  1640155586.69584
# 년 :  2021
# 월 :  12
# 일 :  22
# 시 :  15
# 분 :  46
# 초 :  26
# 마이크로초 :  695840
# 요일 :  2
# 문자열 변환 :  2021-12-22 15:46:26

class xTimer:
    def __init__(self, log_path_file = None, append = True, std_out=True) -> None:
        self.time_start = None
        self.time_before = None
        self.logger = None
        
        if is_str(log_path_file) and len(log_path_file) > 0:
            self.logger = xlogger(log_path_file, append, std_out)
        
    def start(self, *logs):
        self.time_start = time.time()
        self.time_before =  self.time_start
        
        now = datetime.now()
        time_log = f'{str(now)[0:23]} [{round(0.0, 3):.3f}|{round(0.0, 3):.3f}]'
        str_log_list = [str(log) for log in logs]
        
        logs_str = time_log + ' '.join(str_log_list)
        
        if self.logger is not None:
            self.logger.log(logs_str)
        
        return self
        
    def mark(self, proc_name, *logs):
        time_mark = time.time()
        
        self.time_start = self.time_start if self.time_start else time_mark
        time_proc = time_mark - self.time_before if self.time_before else 0.000
        time_elasped = time_mark - self.time_start  if self.time_start  else 0.000
        
        # 시간차이를 구했으면 혅재 로그 시간을 before 타임으로 세팅
        self.time_before = time_mark
        
        now = datetime.now()
        time_log = f'{str(now)[0:23]} [{round(time_elasped, 3):.3f}|{round(time_proc, 3):.3f}] {str(proc_name)}'
        str_log_list = [str(log) for log in logs]
        
        logs_str = time_log + ' '.join(str_log_list)
        if self.logger is not None:
            self.logger.log(logs_str)
        else:
            print(logs_str)
        return self
    
pt_ymmdhms_ms = '^(?P<time>(?P<year>\d{4})-(?P<month>\d{2})-(?P<date>\d{2}) (?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2}).(?P<ms>\d{3}))'
pt_ymmdhms = '^(?P<time>(?P<year>\d{4})-(?P<month>\d{2})-(?P<date>\d{2}) (?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2})'
pt_ymmdhm = '^(?P<time>(?P<year>\d{4})-(?P<month>\d{2})-(?P<date>\d{2}) (?P<hour>\d{2})'
pt_ymmdh = '^(?P<time>(?P<year>\d{4})-(?P<month>\d{2})-(?P<date>\d{2}) (?P<hour>\d{2})'
pt_ymd_h = '^(?P<time>(?P<Y>\d{4})-(?P<M>\d{2})-(?P<d>\d{2}) (?P<H>\d{2})'
pt_ymmdhms_ms_thread = '^(?P<time>(?P<year>\d{4})-(?P<month>\d{2})-(?P<date>\d{2}) (?P<hour>\d{2}):(?P<min>\d{2}):(?P<sec>\d{2}).(?P<ms>\d{3})) \[(?P<thread>[^\]]+?)\]'

@contextmanager
def pool_context(*args, **kwargs):
    pool = multiprocessing.Pool(*args, **kwargs)
    yield pool
    pool.terminate()
    
def get_len(o):
    return 0 if is_none(o) else len(o)
    
    
class HttpRequest:
    def __init__(self, url=None, headers=None, cookies=None) -> None:
        self.headers = None
        self.cookies = None
        self.params = {}
        self.data = {}
        
        if is_not_empty_str(url):
            self.url = url
            
        if is_it(headers):
            self.headers = headers
        
        if is_it(cookies):
            self.cookies = cookies
        
        self.data = {}
        
    def param(self, k, v):
        self.params[k] = v
        return self
    
    def data(self, k, v):
        self.data[k] = v
        return v
        
    def set_params(self, params):
        self.params = params
        return self
    
    def set_headers(self, headers):
        self.headers = headers
        
        return self
    
    def set_cookies(self, cookies):
        self.cookies = cookies
        return self
    
    def set_data(self, data):
        self.data  = data
        return self
    
    def get(self):
        res = None
        if get_len(self.params) > 0 and get_len(self.headers) > 0 and get_len(self.cookies) > 0:
            res.requests.get(self.url, params=self.params, cookies=self.cookies)
        elif get_len(self.params) > 0 and get_len(self.headers) >0:
            res = requests.get(self.url, params=self.params, headers=self.headers)
        elif get_len(self.params) > 0:
            res = requests.get(self.url, params=self.params)
        else:
            res = requests.get(self.url)
        return res
    
    
    def get(self, url=None):
        if is_not_empty_str(url):
            self.url = url
        
        if is_empty_str(self.url):
            print('url is missing')
            return
        
        res = None
        if get_len(self.params) > 0 and get_len(self.headers) > 0 and get_len(self.cookies) > 0:
            res.requests.get(self.url, params=self.params, cookies=self.cookies)
        elif get_len(self.params) > 0 and get_len(self.headers) >0:
            res = requests.get(self.url, params=self.params, headers=self.headers)
        elif get_len(self.params) > 0:
            res = requests.get(self.url, params=self.params)
        else:
            res = requests.get(self.url)
        return res
    
    def post(self, data):
        res = None
        try:
            post_data = data if is_str(data) else orjson.dumps(data)
            res = requests.post(self.url, data = post_data)
        except Exception as e:
            print(f'HttpRequest post exception = {e}')
            
def filter_object(filter_fn = None, o = '', key = 'root', path = '', depth = 0, index = 0):
    path = key if depth == 0 else path + ">" + str(key)
    depth = depth + 1
    
    t = type_(o)
    r = None
    if t == 'dict':
        r = {}
    elif t == 'list':
        r = []
    else:
        t = 'vt'
        r = o
        
        
    if filter_fn is not None:
        filter_fn(key, o, t, depth, path)
        
       
    # filter_fn = None, o = '', key = 'root', path = '', depth = 0, index = 0   
        
    idx = 0
    if t == 'list':
        for x in o:
            k = f'[{idx}]'
            r.append(filter_object(filter_fn, x, k, path, depth, idx))
            idx += 1
    elif t == 'dict':
        for k, v in o.items():
            r[k] = filter_object(filter_fn, v, k, path, depth, idx)
            idx += 1
            
    return r            

