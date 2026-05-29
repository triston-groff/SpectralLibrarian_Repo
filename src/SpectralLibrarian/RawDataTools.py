from .utilities import takeClosest, eic_interpolator_linear_pure, within_ppm, ppm_error

from pyteomics import mzml
from typing import *

import os, datetime, sqlite3
import pandas as pd, numpy as np


def readmzML(mzmldir:str, ppms:List[float], abdcos:List[float], half_width:float=0.65, ce:float=-1):
    # Count the number of MS1 and MS2 spectra
    totalSpecnum = {}
    with mzml.read(mzmldir.replace('"', "")) as reader:
        for spectrum in reader:
            mslevel = spectrum["ms level"]
            if mslevel not in totalSpecnum.keys():
                totalSpecnum[mslevel] = 0
            totalSpecnum[mslevel] += 1

        totalSpec = sum(totalSpecnum.values())
        print("Total Spectrum: " + str(totalSpec))
        for mslevel in list(np.sort(list(totalSpecnum.keys()))):
            print(totalSpecnum[mslevel], " MS" + str(mslevel) + " spectra detected")

    rawfilename = os.path.splitext(os.path.basename(mzmldir))[0]
    dbfiledir = os.path.join(os.path.dirname(os.path.dirname(mzmldir)), "dbfile")
    procob = dbfile(specdb_path=os.path.join(dbfiledir, rawfilename + ".db"), metadb_path=os.path.join(dbfiledir, rawfilename + "_metainfo.db"), ppms=ppms, abdcos=abdcos)

    if procob.readfile:
        procob.create_dbfiles()
        # Read the ms1 spectra (m/z, intensity) in a batch and save that in a dbfile: dataframe(mz, rt, intensity, file)
        ms1_meta, ms2_meta = [], []
        with mzml.read(mzmldir.replace('"', "")) as reader:
            for count, spectrum in enumerate(reader):
                rt = spectrum["scanList"]["scan"][0]["scan start time"]
                rt = np.round(rt / 60 if rt.unit_info.lower() in ["seconds", "second"] else rt, 7)
                if 'positive scan' in spectrum.keys():
                    polarity = 1
                elif 'negative scan' in spectrum.keys():
                    polarity = -1
                else:
                    raise AttributeError("Polarity cannot be found in spectrum : " + spectrum['index'] + ". Please check the mzML file")

                try:
                    n = len(spectrum["m/z array"])
                    ms_data = list(zip([rt]*n, [polarity]*n, spectrum["m/z array"], spectrum["intensity array"]))
                except KeyError:
                    continue

                mslevel = spectrum["ms level"]
                if mslevel == 1:
                    procob.rts_ms1.append(rt)
                    procob.insert_data(ms_data=ms_data, to="spectra_ms1")
                    ms1_meta.append((rt,))
                elif mslevel == 2:
                    procob.rts_ms2.append(rt)
                    procob.insert_data(ms_data=ms_data, to="spectra_ms2")
                    # TODO : in ZT scan, spectrum["precursorList"]["precursor"][0].keys = ['isolationWindow', 'selectedIonList', 'activation']
                    # TODO : carefully take note 'selectedIonList'={'count':1, 'selectedIon': [{'selected ion m/z': 200.671398384595 m/z}]}
                    try:
                        iit = spectrum["scanList"]["scan"][0]["ion injection time"]
                    except:
                        Warning("No ion injection time found in spectrum : " + str(spectrum['index']) + ". Please check the mzML file")
                        iit = 0

                    try:
                        res = spectrum["scanList"]["scan"][0]["mass resolving power"]
                    except:
                        Warning("No mass resolving power found in spectrum : " + str(spectrum['index']) + ". Please check the mzML file")
                        res = 0

                    try:
                        targetMz = spectrum["precursorList"]["precursor"][0]["selectedIonList"]["selectedIon"][0]["selected ion m/z"]  # TODO : Explore MS2 data more
                    except:
                        Warning("No target ion m/z is found in spectrum : " + str(spectrum['index']) + ". Please check the mzML file")
                        targetMz = 0

                    try:
                        iso_start = targetMz - spectrum["precursorList"]["precursor"][0]["isolationWindow"]['isolation window lower offset']
                        if iso_start == 0.0:
                            iso_start = targetMz - half_width
                    except:
                        Warning("No isolation window lower offset is found in spectrum : " + str(spectrum['index']) + " Half size is forced to " + str(half_width) + ". Please check the mzML file")
                        iso_start = targetMz - half_width

                    try:
                        iso_end = targetMz + spectrum["precursorList"]["precursor"][0]["isolationWindow"]['isolation window upper offset']
                        if iso_end == 0.0:
                            iso_end = targetMz + half_width
                    except:
                        Warning("No isolation window upper offset is found in spectrum : " + str(spectrum['index']) + " Half size is forced to " + str(half_width) + ". Please check the mzML file")
                        iso_end = targetMz + half_width

                    try:
                        CE = spectrum["precursorList"]["precursor"][0]["activation"]['collision energy']  # eV
                    except:
                        Warning("No CE is found in spectrum : " + str(spectrum['index']) + ". Please check the mzML file")
                        CE = ce
                    ms2_meta.append((rt, iit, res, iso_start, iso_end, CE))

        procob.insert_data(ms_data=ms1_meta, to="meta_ms1")
        procob.insert_data(ms_data=ms2_meta, to="meta_ms2")
        procob.commit()
        procob.create_indices()

    return procob


class dbfile():
    def __init__(self, specdb_path:str, metadb_path:str, ppms, abdcos, **kwargs):
        self.specdb_path, self.metadb_path = specdb_path, metadb_path
        self.ms1ppm, self.ms2ppm = ppms
        self.ms1abdco, self.ms2abdco = abdcos

        self.readfile = True
        self.timestamp = None
        self.rts_ms1, self.rts_ms2 = [], []
        self.ms2_ms1_map = None
        self.feat_ms2_map = None
        self.peaks = None
        self.num_spec_mapped = None
        self.num_ms2_prec = None
        self.samplename = os.path.splitext(os.path.basename(specdb_path))[0]
        if os.path.exists(self.specdb_path) and os.path.exists(self.metadb_path):
            print(self.specdb_path + " exists. Reading Meta file : " + self.metadb_path)
            conn = sqlite3.connect(self.metadb_path)
            cursor = conn.cursor()
            print("Reading MS1 RTs")
            cursor.execute(''' SELECT rt FROM ms1_meta ''')
            # cursor.execute(''' SELECT rt FROM ms1_meta WHERE (file = COALESCE(?, file))''', [samplename])
            self.rts_ms1 = list(np.unique([rt for rt in cursor.fetchall()]))
            print("Reading MS2 RTs")
            cursor.execute(''' SELECT rt FROM ms2_meta ''')
            # cursor.execute(''' SELECT rt FROM ms2_meta WHERE (file = COALESCE(?, file))''', [samplename])
            self.rts_ms2 = list(np.unique([rt for rt in cursor.fetchall()]))
            conn.close()
            self.readfile = False
        else:
            os.makedirs(os.path.dirname(specdb_path), exist_ok=True)
            os.makedirs(os.path.dirname(metadb_path), exist_ok=True)
            # remove any rows in metaDB to prevent overlapping info
            if os.path.exists(self.metadb_path):
                with sqlite3.connect(self.metadb_path) as conn:
                    cursor = conn.cursor()
                    # cursor.execute("DELETE FROM ms1_meta WHERE file = ?", (samplename,))
                    cursor.execute("DELETE FROM ms2_meta WHERE sample = ?", (self.samplename,))
                    conn.commit()
            # Then read the data
            print("Either spectrum or metafile does not exist")

        print("Connecting to SQLite database for spectrum data:", self.specdb_path)
        print("Connecting to SQLite database for spectrum meta:", self.metadb_path)
        self.conn_spec = sqlite3.connect(self.specdb_path)
        self.conn_meta = sqlite3.connect(self.metadb_path)
        self.cursor_spec = self.conn_spec.cursor()
        self.cursor_meta = self.conn_meta.cursor()

    def create_dbfiles(self):
        self.conn_spec.execute('PRAGMA journal_mode = MEMORY')
        self.conn_spec.execute('PRAGMA synchronous = OFF')
        self.cursor_spec.execute('''CREATE TABLE IF NOT EXISTS ms1_data (rt REAL, polarity INT, mz REAL, intensity REAL)''')
        self.cursor_spec.execute('''CREATE TABLE IF NOT EXISTS ms2_data (rt REAL, polarity INT, mz REAL, intensity REAL)''')
        self.conn_spec.commit()

        self.conn_meta.execute('PRAGMA journal_mode = MEMORY')
        self.conn_meta.execute('PRAGMA synchronous = OFF')
        # self.cursor_meta.execute('''CREATE TABLE IF NOT EXISTS ms2_meta (id INTEGER PRIMARY KEY, sample TEXT, rt REAL, ion_inj_time REAL, resolution REAL, iso_start REAL, iso_end REAL, ce REAL)''')
        self.cursor_meta.execute('''CREATE TABLE IF NOT EXISTS ms1_meta (rt REAL)''')
        self.cursor_meta.execute('''CREATE TABLE IF NOT EXISTS ms2_meta (rt REAL, ion_inj_time REAL, resolution REAL, iso_start REAL, iso_end REAL, ce REAL)''')
        self.conn_meta.commit()

        self.drop_indices()

    def create_indices(self):
        # Generate index for speed up in searching
        self.conn_spec.execute('PRAGMA journal_mode = MEMORY')  # save transaction log (journal) into memory for speedup instead of disk
        self.conn_spec.execute('PRAGMA synchronous = OFF')  # turn off disk synchronization
        self.conn_spec.execute('PRAGMA cache_size = -100000')   # Adjust cache size 100,000 page in KB -> 100MB
        self.cursor_spec.execute('CREATE INDEX IF NOT EXISTS idx_rt_ms1 ON ms1_data(rt)')  # make index for rt column in the table 'data'
        self.cursor_spec.execute('CREATE INDEX IF NOT EXISTS idx_mz_ms1 ON ms1_data(mz)')  # make index for mz column in the table 'data'
        self.cursor_spec.execute('CREATE INDEX IF NOT EXISTS idx_rt_ms2 ON ms2_data(rt)')  # make index for rt column in the table 'data'
        self.cursor_spec.execute('CREATE INDEX IF NOT EXISTS idx_mz_ms2 ON ms2_data(mz)')  # make index for mz column in the table 'data'

        self.conn_meta.execute('PRAGMA journal_mode = MEMORY')  # save transaction log (journal) into memory for speedup instead of disk
        self.conn_meta.execute('PRAGMA synchronous = OFF')  # turn off disk synchronization
        self.conn_meta.execute('PRAGMA cache_size = -100000')   # Adjust cache size 100,000 page in KB -> 100MB
        self.cursor_meta.execute('CREATE INDEX IF NOT EXISTS idx_rt_ms1meta ON ms1_meta(rt)')  # make index for rt column in the table 'data'
        self.cursor_meta.execute('CREATE INDEX IF NOT EXISTS idx_rt_ms2meta ON ms2_meta(rt)')  # make index for rt column in the table 'data'
        self.commit()

    def drop_indices(self):
        # delete index if exists to speed up inserting
        self.conn_spec.execute('PRAGMA journal_mode = MEMORY')
        self.conn_spec.execute('PRAGMA synchronous = OFF')
        self.cursor_spec.execute('DROP INDEX IF EXISTS idx_rt_ms1')
        self.cursor_spec.execute('DROP INDEX IF EXISTS idx_mz_ms1')
        self.cursor_spec.execute('DROP INDEX IF EXISTS idx_rt_ms2')
        self.cursor_spec.execute('DROP INDEX IF EXISTS idx_mz_ms2')
        self.conn_meta.execute('PRAGMA journal_mode = MEMORY')
        self.conn_meta.execute('PRAGMA synchronous = OFF')
        self.cursor_meta.execute('DROP INDEX IF EXISTS idx_rt_ms1meta')
        self.cursor_meta.execute('DROP INDEX IF EXISTS idx_rt_ms2meta')
        self.commit()

    def commit(self):
        self.conn_spec.commit()
        self.conn_meta.commit()

    def close(self):
        self.conn_spec.close()
        self.conn_meta.close()

    def insert_data(self, ms_data, to:Literal["spectra_ms1", "spectra_ms2", "meta_ms1", "meta_ms2"]):
        if to == "spectra_ms1":
            self.cursor_spec.executemany("INSERT INTO ms1_data (rt, polarity, mz, intensity) VALUES (?, ?, ?, ?)", ms_data)
            self.conn_spec.commit()
        elif to == "spectra_ms2":
            self.cursor_spec.executemany("INSERT INTO ms2_data (rt, polarity, mz, intensity) VALUES (?, ?, ?, ?)", ms_data)
            self.conn_spec.commit()
        elif to == "meta_ms1":
            self.cursor_meta.executemany("INSERT INTO ms1_meta (rt) VALUES (?)", ms_data)
            self.conn_meta.commit()
        elif to == "meta_ms2":
            self.cursor_meta.executemany("INSERT INTO ms2_meta (rt, ion_inj_time, resolution, iso_start, iso_end, ce) VALUES (?, ?, ?, ?, ?, ?)", ms_data)
            self.conn_meta.commit()
        else:
            raise NotImplementedError("Inserting data into other than 'spectra_ms1', 'spectra_ms2', or 'meta_ms1', 'meta_ms2' tables is not implemented")

    def getEICinfo(self, rtRange, precursor_mz=None):
        rts, ints = self.extractEIC(rt_start=rtRange[0], rt_end=rtRange[1], mz=precursor_mz)
        try:
            apex_idx = np.argmax(ints)
            mz_highs, high_ints = [], []
            for shift in list(range(-5, 6)):
                try:
                    mz_high, high_int = self.getApexMz(rt=rts[apex_idx + shift], mz=precursor_mz)
                    mz_highs.append(mz_high)
                    high_ints.append(high_int)
                except IndexError as ie:
                    continue
            ms1_ppm_practical = np.max(ppm_error(np.array(mz_highs)[1:], np.array(mz_highs)[:-1])) if len(mz_highs) > 1 else None
        except ValueError as ve:
            ms1_ppm_practical = None

        return rts, ints, ms1_ppm_practical

    def extractEIC(self, rt_start, rt_end, mz, dmz=None):
        """Given m/z and RT range, consider m/z ppm and get the EIC by summing the intensity over mz range for each RT"""
        if dmz is None:
            mz_start, mz_end = mz*(1 - self.ms1ppm/1e6), mz*(1 + self.ms1ppm/1e6)
        else:
            mz_start, mz_end = mz - dmz, mz + dmz

        with sqlite3.connect(self.specdb_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT rt, mz, intensity FROM ms1_data WHERE mz >= ? AND mz <= ? AND rt >= ? AND rt <= ? AND intensity > ?", (mz_start, mz_end, rt_start, rt_end, self.ms1abdco))

            # fetch the next 10000 size of results every each round and save the results by summing the intensity for each RT
            filtered_set_rts = list(np.sort([x for x in self.rts_ms1 if rt_start <= x <= rt_end]))
            intensity_by_rt = {rt: 0 for rt in filtered_set_rts}
            while True:
                results = cursor.fetchmany(10000)
                if not results:
                    break

                for rt, mz_value, intensity in results:
                    intensity_by_rt[rt] += intensity

            intensities = [intensity_by_rt[rt] for rt in filtered_set_rts]

        return filtered_set_rts, intensities

    def getTIC(self, rtRange=None):
        rt_start, rt_end = rtRange if not (rtRange is None) else self.rts_ms1[0], self.rts_ms1[-1]
        rts_search = [rt for rt in self.rts_ms1 if (rt_start <= rt <= rt_end)]
        tics = []
        for rt in rts_search:
            print("Extracting MS1 spectrum from rt = " + str(rt))
            spectrum = self.getSpectrum(rt=rt, mslevel=1)
            tics.append(np.sum(spectrum[:,1]))
        return rts_search, tics

    def getSpectrum(self, rt, mslevel, inthre=0.0, iit=None):
        """Retrieve list of tuples (mz, intensity) for the specified retention time."""
        conn = sqlite3.connect(self.specdb_path)
        cursor = conn.cursor()
        if mslevel == 1: # return MS1 spectrum at the given RT (rt)
            # print("Extracting MS1 spectrum at RT = " + str(rt))
            query = '''SELECT mz, intensity FROM ms1_data WHERE rt = ? AND intensity > ?'''
            cursor.execute(query, (rt, inthre))
            mzs, abds = [], []
            for mz, abd in cursor.fetchall():
                mzs.append(mz)
                abds.append(abd)
            spectrum = np.transpose(np.array([mzs, abds]))
        elif mslevel == 2: # return the MS2 spectrum at the given RT (rt), obtained for the target mz (mz_target)
            # print("Extracting MS2 spectrum at RT = " + str(rt))
            query = '''SELECT mz, intensity FROM ms2_data WHERE rt = ? AND intensity > ?'''
            cursor.execute(query, (rt, inthre))
            mzs, abds = [], []
            for mz, abd in cursor.fetchall():
                mzs.append(mz)
                if not (iit is None): # ion injection time should be given, if the spectrum intensity unit is not the 'CPS (count per second)' but just the 'count'
                    abds.append(abd/iit)
                else:
                    abds.append(abd)
            spectrum = np.transpose([mzs, abds])
        else:
            raise NotImplementedError("mslevel must be 1 or 2")
        conn.close()
        return spectrum

    def getMS2Metadata(self, rtRange: list, mz_target=None):
        """
        Retrieve metadata (isolation window, CE) of the spectrum for the specified retention time and mz target of the MS2 spectrum.
        :param mz_target: float, m/z target of MS2 spectrum
        :return: pd.DataFrame each row containing isolation window and collision energy (CE) for all the ms2scans meeting the requirement
        """
        conn = sqlite3.connect(self.metadb_path)
        cursor = conn.cursor()
        samplename = os.path.splitext(os.path.basename(self.specdb_path))[0]
        # amax >= bmin and bmax >= amin
        if len(rtRange) == 1:
            if mz_target is not None:
                query = "SELECT rt, ion_inj_time, resolution, iso_start, iso_end, ce FROM ms2_meta WHERE rt = ? AND (iso_start <= ?) AND (iso_end >= ?)"
                cursor.execute(query, (rtRange[0], mz_target * (1 + self.ms1ppm / 1e6), mz_target * (1 - self.ms1ppm / 1e6)))
            else:
                query = "SELECT rt, ion_inj_time, resolution, iso_start, iso_end, ce FROM ms2_meta WHERE rt = ?"
                cursor.execute(query, (rtRange[0]))
        else:
            if mz_target is not None:
                query = "SELECT rt, ion_inj_time, resolution, iso_start, iso_end, ce FROM ms2_meta WHERE rt >= ? AND rt <= ? AND (iso_start <= ?) AND (iso_end >= ?)"
                cursor.execute(query, (rtRange[0], rtRange[1], mz_target * (1 + self.ms1ppm / 1e6), mz_target * (1 - self.ms1ppm / 1e6)))
            else:
                query = "SELECT rt, ion_inj_time, resolution, iso_start, iso_end, ce FROM ms2_meta WHERE rt >= ? AND rt <= ?"
                cursor.execute(query, (rtRange[0], rtRange[1]))
        results = pd.DataFrame.from_records(cursor.fetchall(), columns=[des[0] for des in cursor.description])
        conn.close()
        return results

    def getApexMz(self, rt, mz):
        """Given m/z and RT value, consider m/z ppm and get the highest-intensity m/z and the corresponding intensity"""
        # The index of the closest retention time (rt) using the takeClosestInd method
        closest_rt = self.rts_ms1[takeClosest(self.rts_ms1, rt)[0]]
        mz_start, mz_end = mz*(1 - self.ms1ppm/1e6), mz*(1 + self.ms1ppm/1e6)

        result = self.cursor_spec.execute("SELECT mz, intensity FROM ms1_data WHERE rt = ? AND mz >= ? AND mz <= ? ORDER BY intensity DESC LIMIT 1", [closest_rt, mz_start, mz_end]).fetchone()
        if result:
            return result
        else:
            return mz, 0

    def gatherMS2spec(self, rtRange, precursor_mz, ce, mzs_to_exclude=None):
        ms1_rts, ms1_eic, ppmerr = self.getEICinfo(rtRange=rtRange, precursor_mz=precursor_mz)
        ms2_meta = self.getMS2Metadata(rtRange=rtRange, mz_target=precursor_mz)
        ms2meta_ce = ms2_meta.loc[ms2_meta["ce"] == ce]
        s = eic_interpolator_linear_pure(ms1_rts, ms1_eic)

        ms1abds, iits, ms2spectra, ms2rts = [], [], [], []
        for i, meta in ms2meta_ce.iterrows():
            ms2_rt, iit = meta["rt"], meta["ion_inj_time"]
            ms2_spectrum = self.getSpectrum(rt=ms2_rt, mslevel=2, inthre=self.ms2abdco)  # Normalize by ion injection time
            if len(ms2_spectrum) == 0:
                Warning("MS2 spectrum is empty. This must indicate the MS2 spectrum has only peaks with intensity less than 'absint', or the MS2 RT does not match exactly to that in sqlite3 dbfile.")
                continue
            else:
                if np.min(ms1_rts) < ms2_rt < np.max(ms1_rts):  # only when precursor abundance can be inferred
                    ms1abds.append(s(ms2_rt))
                    iits.append(iit)
                    if not (mzs_to_exclude is None):
                        for mz_exclude in mzs_to_exclude[1:]:
                            ms2_spectrum = ms2_spectrum[np.where(~within_ppm(ms2_spectrum[:,0], mz_exclude, self.ms2ppm))[0], :]
                        ms2spectra.append(ms2_spectrum)
                    else:
                        ms2spectra.append(ms2_spectrum)
                    ms2rts.append(ms2_rt)
        return ms1abds, iits, ms2spectra, ms2rts


__all__ = ["readmzML", "dbfile"]