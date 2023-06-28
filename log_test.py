from codecs import getreader
from distutils import file_util
from doctest import DocFileSuite
from re import X
from string import hexdigits
import sys
from time import sleep
from timeit import repeat

sys.path.insert(0, "/")
from util.xutil import *
import datetime
import linecache as lc
from pyarrow import json


# # Using linecache.getline() method
# gfg = lc.getline('File.txt', 1)
# print(gfg)

def line_to_hash_write(data_path_file, target_path, line_indexer, hash_cnt=10):
    file_name_without_ext = file_only_but_extension(data_path_file)
    file_name_default = file_only(data_path_file)
    defaut_extension = extension_only(file_name_default)

    # 미리 파일 객체를 생성하여 두고 사용하자
    file_name_list = [make_file_name(target_path,
                                     file_name_without_ext + '_' + f'{x:{0}{digit_len(hash_cnt)}}' + '.' + defaut_extension)
                      for x in range(hash_cnt)]

    fw_list = [xFileWriter(x) for x in file_name_list]

    def line_stream_to_ex_hash_file(line_stream):
        timer = xTimer().mark('line_stream_to_hash_file start')
        line_cnt = 0
        ex_uno_has_cnt = 0
        for line in line_stream:
            line_idxed = line_indexer(line)
            if line_idxed < 0:
                ex_uno_has_cnt = ex_uno_has_cnt + 1
            else:
                hash_value = line_idxed % hash_cnt
                fw = fw_list[hash_value]
                fw.write_line(line, False) if fw else 1
                line_cnt += 1
                if line_cnt % 100_0000 == 0:
                    timer.mark(f'line_cnt : {line_cnt}', line_idxed)

        [fp.close() for fp in fw_list]
        timer.mark('line_stream_to_hash_file end', 'ex_uno_has_cnt : ', ex_uno_has_cnt)

    return line_stream_to_ex_hash_file


def convert_file(path_file, hashed_path):
    timer = xTimer().start()

    file_only = file_only_but_extension(path_file)

    # dictionary stream으로 부터 필요한 부분을 추출
    stream_indexer = index_(
        lambda x: int(x['uno'])
        , lambda x: int(x['date'][8:])  # 2021-04-01
        , lambda x: [str(x['count']) + '.' + str(x['songid']) for x in x['items']]
    )

    dict_stream = file_to_dict_stream(path_file)
    indexed_stream = stream_indexer(dict_stream)

    # uno 기준으로 합치기
    uno_dict = {}
    idx = 0
    for v in indexed_stream:
        uno = v[0]
        date = v[1]
        item_list = ','.join([str(x) for x in v[2]])

        # 신규 uno 이면 일 슬롯(slot, 31개)를 만든다
        if uno not in uno_dict:
            uno_dict[uno] = ['' for x in range(31)]

        uno_dict[uno][date - 1] = item_list

    converted_file = path_file_ext(hashed_path, file_only, ".uno")
    fw = xFileWriter(converted_file)
    idx = 0
    for uno, items in uno_dict.items():
        # 각 일자별 아이템들을 join('|') 한다
        line = str(uno) + "^" + '|'.join(items)
        fw.write_line(line)
        idx += 1
    fw.close()
    timer.mark(f'file={converted_file}  completed')


def make_uno_parquet(path_file, hashed_path):
    timer = xTimer().start()

    file_only = file_only_but_extension(path_file)
    stream_indexer = index_(
        lambda x: int(x['uno'])
        , lambda x: int(x['date'][8:])  # 2021-04-01
        , lambda x: [str(x['count']) + '.' + str(x['songid']) for x in x['items']]
    )

    dict_stream = file_to_dict_stream(path_file)
    indexed_stream = stream_indexer(dict_stream)

    # uno 기준으로 합치기
    uno_dict = {}
    uidx = 0
    first = True
    for v in indexed_stream:
        uno = v[0]
        date = v[1]
        item_list = ','.join([str(x) for x in v[2]])

        # 신규 uno 이면 일 슬롯(slot, 31개)를 만든다
        if uno not in uno_dict:
            uno_dict[uno] = ['' for x in range(31)]
            uidx += 1

        # 일 slot에 데이터를  넣는다
        uno_dict[uno][date - 1] = item_list

    uno_list = []
    item_list = []

    # (uno, item) list 추출
    ridx = 0
    for uno, items in uno_dict.items():
        items = '|'.join(items)
        uno_list.append(uno)
        item_list.append(items)
        ridx += 1

    uno_item_dict = {
        'uno': uno_list,
        'item': item_list
    }

    parquet_path_file = path_file_ext(hashed_path, file_only, '.parquet')
    df = pd.DataFrame(uno_item_dict)
    df.to_parquet(parquet_path_file)

    timer.mark(f'result ---------------->[{uidx}, {ridx} {parquet_path_file}')
    return uno_item_dict


def file_group_read(file_list, hashed_path):
    result = []
    for file in file_list:
        r = make_uno_parquet(file, hashed_path)
        result.append(r)

    return result


def get_uno_ex(uno_ex_dict):
    def get_uno_from(x):
        uno = int(orjson.loads(x)['uno'])
        # expred uno를 제거하기 위해 uno_ex_dict 에 있는지 검사
        if uno in uno_ex_dict.keys():
            uno_ex_dict[uno] = uno_ex_dict[uno] + 1
            uno = -1
        else:
            uno_ex_dict[uno] = 1
        return uno

    return get_uno_from


# 파일을 UNO별로 나누어서 저장한다 (병렬처리를 위해서)
def split_by_uno(hash_count, data_path_file, splitted_path):
    """월데이터(2020_01.json)를 hash count 만큼 분리하여 병렬처리한다
    Args:
        hash_count (bool): uno를 hash(uno / hash_count)
        data_path_file (_type_): 월 데이터 파일 path+file 
        splitted_path (_type_): 분할한 path
    """
    # uno 추출 함수
    get_uno = lambda x: int(orjson.loads(x)['uno'])

    # expred 된 uno를 제거한다
    ex_file_path_name = 'D:/data/geine_time/ex_uno/ex_uno_tot.txt'
    uno_ex_dict = {}
    for line in line_stream_from_file(ex_file_path_name):
        ex_uno = line.split('\t')[0]
        if ex_uno in uno_ex_dict.keys():
            uno_ex_dict[ex_uno] = uno_ex_dict[ex_uno] + 1
            print('has key', uno_ex_dict[ex_uno])
        else:
            uno_ex_dict[ex_uno] = 1

    print('ex_uno_cnt', len(uno_ex_dict))

    # stream에서 uno를 hash (uno / hash_count)해서 분리로직
    line_hash_writer = line_to_hash_write(data_path_file, splitted_path, get_uno_ex(uno_ex_dict), hash_count)

    # 파일을 stream으로 읽는다
    line_stream = line_stream_from_file(data_path_file, False)

    # hash된 값을 파일로 쓴다
    line_hash_writer(line_stream)


def main():
    # totl 시간을 처리하기 위한 타이머
    tot_timer = xTimer().start()

    # 월 데이터를 여러개의 파일로 나눈다 > 병렬처리
    hash_count = 1000

    # 처리할 파일 (to-do 사용자 interface로 부터 경로_파일명을 입력할 수 있도록하자)
    data_path_file = 'D:/data/log/timemachine/timedependent/2020/01/2020_01.json'
    file = file_only_but_extension(data_path_file)
    path = path_only(data_path_file)

    # 병렬처리를 위한 분할폴더, 작업폴더 생성
    splitted_path = mk_dir(path, 'ex_splitted_json')
    hashed_path = mk_dir(path, 'ex_hashed_parquet')
    merged_path = mk_dir(path, 'ex_merged_parquet')

    # 파일 분할 
    split_by_uno(hash_count, data_path_file, splitted_path)

    # multi process는 CPU CORE 베이스이기 때문에 cpu_count 만큼 병렬처리 갯수를 정한다
    proc_count = multiprocessing.cpu_count()

    file_pattern = '*'
    files = file_path_list(splitted_path, file_pattern)
    tot_timer.mark('file-list', len(files))

    # 분할된 파일을 proc_count(cpu_count)로 나누어서 cpu(core)갯수로 나누어서 처리
    file_group = group_by_cnt(proc_count, files)  # 파일을

    # process에 여러개인자(multiple arguments)를 넣기 위한 util
    zipped_argument = arg_zip(file_group, hashed_path)

    # 분할된 파일 병렬처리
    with multiprocessing.Pool(processes=proc_count) as pool:
        results = pool.starmap(file_group_read, zipped_argument)
        tot_timer.mark('post end ----------->1', f'tot result = {len(results)}')

        # post process, 병렬처리 종료 후 월별 데이터를 하나로 모은 parquet를 생성한다
        monthly_dict_len = 0
        df_list = []
        for uno_item_dict_list in results:
            for uno_item_dict in uno_item_dict_list:
                dict_len = len(uno_item_dict['uno'])
                monthly_dict_len = monthly_dict_len + dict_len
                df = pd.DataFrame(uno_item_dict)
                df_list.append(df)

        # 월별 하나의 dataframe 생성
        monthly_df = pd.concat(df_list)
        tot_timer.mark('post proc ----------->2',
                       f'monthly_dict_len={monthly_dict_len}   monthly_df shape = {monthly_df.shape}')

        # data를 sliceing 하기 위해 f(ile)id column을 생성한다
        monthly_df['fid'] = monthly_df.apply(lambda row: row['uno'] % hash_count, axis=1)
        tot_timer.mark('post proc ----------->3', f'tot_mon_parquet={monthly_df.shape}')

        print(df.head(1))

        # 월 dataframe을 parquet로 저장한다
        monthly_parquet = path_file_ext(merged_path, file, '.parquet')
        monthly_df.to_parquet(monthly_parquet)
        tot_timer.mark('post proc ----------->4', f'tot_mon_parquet={monthly_parquet}')

    # 최종시간
    tot_timer.mark('multiprocessing.Pool END ========================================>')


def get_value_by_uno(uno, hash_count):
    path = 'D:/data/log/timemachine/timedependent/2020/01/hashed'
    str_expr = f'uno == {uno}'  # 나이가 10 이다 (비교연산자 ==)

    hasehed = uno % hash_count
    post_fix = f'{hasehed:{0}{digit_len(hash_count)}}'
    path_file = path_file_ext(path, '2020_01_' + post_fix, '.parquet')
    print('file===>', path_file)

    file_name = file_only(path_file)
    yyyy = file_name[0:4]
    mm = file_name[5:7]

    try:
        df = pd.read_parquet(path_file)
    except Exception as e:
        print(e)
        return None

    df_q = df.query(str_expr)
    rowcnt, colcnt = df_q.shape
    print(f'rowcnt={rowcnt}, colcnt={colcnt}')

    result_list = []

    if rowcnt == 1 and colcnt == 2:
        uno = df_q.iloc[0, 0]
        items_list = df_q.iloc[0, 1].split('|')

        day_in_month = 0

        for date_items in items_list:
            day_in_month += 1
            if len(date_items) > 0:
                date = f'{yyyy}-{mm}-{day_in_month:02}'
                date_item_list = {'uno': str(uno), 'date': date, 'items': []}

                date_items_list = date_items.split('|')
                for count_song_list in date_items_list:
                    for count_song in count_song_list.split(','):
                        splitted = count_song.split('.')
                        count_song = {
                            'count': int(splitted[0]),
                            'songid': splitted[1]
                        }

                        date_item_list['items'].append(count_song)

                result_list.append(date_item_list)

        return result_list


def yield_test(cnt):
    idx = 0
    for idx in range(0, cnt):
        yield idx

    # for()
    # yield 1;    
    # yield 2;    
    # yield 3;    
    # yield 4;    
    # yield 5;    
    # yield 6;    
    # yield 7;    


# def egrep(lines, expr):
#     ptc = re.compile(expr)
#     for line in lines:
#         if ptc.search(line):
#             yield line

def egrepclosure(*exprs):
    ptcs = [re.compile(expr) for expr in exprs]

    def egrep(lines):
        for line in lines:
            # if any(ptc.search(line) for ptc in ptcs):
            #     yield line
            if all(ptc.search(line) for ptc in ptcs):
                yield line

    return egrep


def text_to_dict(lines):
    for line in lines:
        try:
            r = orjson.loads(line)
            yield r
        except Exception as e:
            print(e)
            #프로그램 멈추지 않게 하기 위해서



def extract_to_dict(dicts, *keys):
    for dict in dicts:
        val_list = []
        for key in keys:
            val_list.append(dict[key])

        yield val_list


def fgrepclosure(substring):
    def fgrep(lines):
        for line in lines:
            if substring in line:
                yield line

    return fgrep


if __name__ == '__main__':
    file_stream = file_path_stream('./test_log', ".log$")

    pattern = 'lyrics'
    patten2 = 'video'
    egrep = egrepclosure('radio', 'videos', 'artists', '327225965')

    lines = line_stream(file_stream)
    dicts = text_to_dict(lines)
    for dict in egrep(dicts):
        print(dict)
    # for line in egrep(lines):
    #     print(line)
    # dicts = text_to_dict(lines)
    paths = extract_to_dict(dicts, 'uno', 'path')

    # dataList  = {
    #     '01' : [],
    #     '02' : [],
    #  }
    #
    # for uno_path in paths:
    #     print(uno_path)
    #
    # for k, v in dataList.entris():
    #     k, v,

    # grped_line = egrep(lines, pattern)
    # grped_line2 = egrep(grped_line, patten2)

    # for line in grped_line2:
    #     print(line)

    # print(type(gen))

    # timer = xTimer().start()

    # main()

    # timer.mark('load monthly_parque')

    # pq_file_list = file_path_list('D:/data/log/timemachine/timedependent/2020/01/hashed', '.parquet')
    # tot_pq_file = 'D:/data/log/timemachine/timedependent/2020/01/merged/2020_01.parquet'
    # tot_df = pd.read_parquet(tot_pq_file)

    # arr = tot_df.to_numpy()
    # timer.mark('numpy arr', f'tot_df size = {arr.shape}')

    # hash_count = 1000
    # idx = 0
    # timer.start()

    # for i in range(1):
    #     # uno = choice(tot_uno_list)
    #     uno = 310686146
    #     timer.start()
    #     res = get_value_by_uno(uno, hash_count)
    #     timer.mark('res', f'uno->{uno}')

    #     for date_items in res:
    #         date = date_items['date']
    #         items = date_items['items']
    #         print(f'date==============>:{date}')
    #         for item in items:
    #             print(item)

    # [pd.read_parquet(f) for pq_file in pq_file_list]

    # #totl 시간을 처리하기 위한 타이머
    # tot_timer = xTimer().start()

    # #월 데이터를 여러개의 파일로 나눈다 > 병렬처리
    # hash_count = 1000

    # #처리할 파일 (to-do 사용자 interface로 부터 경로_파일명을 입력할 수 있도록하자)
    # data_path_file = 'D:/data/log/timemachine/timedependent/2020/01/2020_01.json'
    # file = file_only_but_extension(data_path_file)
    # path = path_only(data_path_file)

    # #병렬처리를 위한 분할폴더, 작업폴더 생성 
    # splitted_path = mk_dir(path, 'splitted')
    # hashed_path = mk_dir(path, 'hashed')
    # merged_path = mk_dir(path, 'merged')

    # # 파일 분할 
    # split_by_uno(hash_count, data_path_file, splitted_path)

    # # multi process는 CPU CORE 베이스이기 때문에 cpu_count 만큼 병렬처리 갯수를 정한다
    # proc_count = multiprocessing.cpu_count()

    # file_pattern = '*'
    # files = file_path_list(splitted_path, file_pattern)
    # tot_timer.mark('file-list', len(files))

    # #분할된 파일을 proc_count(cpu_count)로 나누어서 cpu(core)갯수로 나누어서 처리
    # file_group = group_by_cnt(proc_count, files) #파일을 

    # #process에 여러개인자(multiple arguments)를 넣기 위한 util
    # zipped_argument = arg_zip(file_group, hashed_path)

    # # 분할된 파일 병렬처리
    # with multiprocessing.Pool(processes = proc_count) as pool:
    #     results = pool.starmap(file_group_read, zipped_argument)
    #     tot_timer.mark('post end ----------->1', f'tot result = {len(results)}')

    #     #post process, 병렬처리 종료 후 월별 데이터를 하나로 모은 parquet를 생성한다
    #     monthly_dict_len = 0
    #     df_list = []
    #     for uno_item_dict_list in results:
    #         for uno_item_dict in uno_item_dict_list:
    #             dict_len = len(uno_item_dict['uno'])
    #             monthly_dict_len = monthly_dict_len + dict_len
    #             df = pd.DataFrame(uno_item_dict)
    #             df_list.append(df)

    #     # 월별 하나의 dataframe 생성
    #     monthly_df = pd.concat(df_list)
    #     tot_timer.mark('post proc ----------->2', f'monthly_dict_len={monthly_dict_len}   monthly_df shape = {monthly_df.shape}')

    #     # data를 sliceing 하기 위해 f(ile)id column을 생성한다
    #     monthly_df['fid'] = monthly_df.apply(lambda row : row['uno'] % hash_count, axis=1)
    #     tot_timer.mark('post proc ----------->3', f'tot_mon_parquet={monthly_df.shape}')

    #     print(df.head(1))

    #     # 월 dataframe을 parquet로 저장한다
    #     monthly_parquet = path_file_ext(merged_path, file, '.parquet')
    #     monthly_df.to_parquet(monthly_parquet)
    #     tot_timer.mark('post proc ----------->4', f'tot_mon_parquet={monthly_parquet}')

    # #최종시간
    # tot_timer.mark('multiprocessing.Pool END ========================================>')

# 로컬에서 테스트한 시간
# 2022-08-30 13:00:10.309 [591.846|591.846] multiprocessing.Pool END ========================================> 
# 2022-08-30 14:44:47.757 [707.332|707.332] multiprocessing.Pool END ========================================> parquet
# 2022-08-30 15:29:35.411 [694.408|694.408] multiprocessing.Pool END ========================================> file < 12M
# 2022-08-30 21:25:34.193 [2946.736|2946.736] multiprocessing.Pool END ========================================> 압축 99개 너무 늦는데????????
# 2022-08-31 10:12:41.226 [679.381|43.646] post end ----------->3 tot_mon_parquet=D:/data/log/timemachine/timedependent/2020/01/hashed/2020_01.parquet
# 2022-08-31 11:01:13.387 [35.814|0.000] multiprocessing.Pool END ========================================> parquet laod
# 2022-08-31 15:46:37.003 [1303.288|0.083] multiprocessing.Pool END ========================================>
