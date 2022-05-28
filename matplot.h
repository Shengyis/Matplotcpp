#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <string>
#include <initializer_list>
#include <tuple>
#include <map>
#include <Python.h>
#include <numpy/arrayobject.h>

namespace plt
{
    static bool does_load_pyplot = 0;
    static PyObject* plt;
    static PyObject* np;

    void load_pyplot()
    {
        if (!does_load_pyplot)
        {
            Py_Initialize();
            plt = PyImport_ImportModule("matplotlib.pyplot");
            np = PyImport_ImportModule("numpy");
            _import_array();
            does_load_pyplot = 1;
        }
    }

    PyObject* getPltFun(const char* argv)
    {
        return PyObject_GetAttrString(plt, argv);
    }

    PyObject* getNpFun(const char* argv)
    {
        return PyObject_GetAttrString(np, argv);
    }

    template <typename T>
    auto eval(const T& x) { return x.eval(); }
    /* template <typename T>
    auto eval(const std::initializer_list<T>& x) { return x; } */
    template <>
    auto eval(const double& x) { return x; }
    template <>
    auto eval(const int& x) { return x; }

    template <typename T>
    PyObject* getPyData(const T& v)
    {
        npy_intp rows = v.rows();
        npy_intp cols = v.cols();
        npy_intp dim[] = { cols, rows };
        int nd = (cols == 1 || rows == 1) ? 1 : 2;
        int dim_start = (nd == 2 || rows == 1) ? 0 : 1;
        PyObject* pArray = PyArray_SimpleNewFromData(nd, dim + dim_start, NPY_DOUBLE, const_cast<double*>(v.data()));
        return pArray;
    }
    template <>
    PyObject* getPyData(const std::initializer_list<double>& v)
    {
        npy_intp dim[] = { (npy_intp)v.size() };
        int nd = 1;
        PyObject* pArray = PyArray_SimpleNewFromData(nd, dim, NPY_DOUBLE, const_cast<double*>(v.begin()));
        return pArray;
    }
    template <>
    PyObject* getPyData(const double& v)
    {
        PyObject* pArray = PyFloat_FromDouble(v);
        return pArray;
    }
    template <>
    PyObject* getPyData(const int& v)
    {
        PyObject* pArray = PyLong_FromLong(v);
        return pArray;
    }

    template <typename Tx, typename Ty>
    PyObject* getPlotArgs(const Tx& x, const Ty& y, const std::string& str = "")
    {
        PyObject* pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, getPyData(x));
        PyTuple_SetItem(pArgs, 1, getPyData(y));
        PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(str.c_str()));
        return pArgs;
    }

    template <typename Tx, typename Ty, typename Tz, typename Tlvl = int>
    PyObject* getContourArgs(const Tx& x, const Ty& y, const Tz& z, const Tlvl& lvl = 10)
    {
        PyObject* pArgs = PyTuple_New(4);
        PyTuple_SetItem(pArgs, 0, getPyData(x));
        PyTuple_SetItem(pArgs, 1, getPyData(y));
        PyTuple_SetItem(pArgs, 2, getPyData(z));
        PyTuple_SetItem(pArgs, 3, getPyData(lvl));
        return pArgs;
    }

    PyObject* getKwargs(const std::map<std::string, std::string>& key)
    {
        PyObject* pKwargs = PyDict_New();
        for (auto const& item : key)
            PyDict_SetItem(pKwargs, PyUnicode_FromString(item.first.c_str()), PyUnicode_FromString(item.second.c_str()));
        return pKwargs;
    }

    template <typename Tx, typename Ty>
    void plot(const Tx& x, const Ty& y, const std::string str = "", const std::map<std::string, std::string>& key = {})
    {
        auto xx = eval(x);
        auto yy = eval(y);
        PyObject* args = getPlotArgs(xx, yy, str);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("plot"), args, kwargs);
    }

    template <typename Tx, typename Ty>
    void plot(const Tx& x, const Ty& y, const std::map<std::string, std::string>& key)
    {
        plot(x, y, "", key);
    }

    template <typename Tx, typename Ty, typename Tz, typename Tlvl = int>
    void contourf(const Tx& x, const Ty& y, const Tz& z, const Tlvl& lvl = 10, const std::map<std::string, std::string>& key = {})
    {
        auto xx = eval(x);
        auto yy = eval(y);
        auto zz = eval(z);
        PyObject* args = getContourArgs(xx, yy, zz, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contourf"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contourf(const Tx& x, const Ty& y, const Tz& z, const std::initializer_list<double>& lvl, const std::map<std::string, std::string>& key = {})
    {
        auto xx = eval(x);
        auto yy = eval(y);
        auto zz = eval(z);
        PyObject* args = getContourArgs(xx, yy, zz, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contourf"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contourf(const Tx& x, const Ty& y, const Tz& z, const std::map<std::string, std::string>& key)
    {
        contourf(x, y, z, 10, key);
    }

    template <typename Tx, typename Ty, typename Tz, typename Tlvl = int>
    void contour(const Tx& x, const Ty& y, const Tz& z, const Tlvl& lvl = 10, const std::map<std::string, std::string>& key = {})
    {
        auto xx = eval(x);
        auto yy = eval(y);
        auto zz = eval(z);
        PyObject* args = getContourArgs(xx, yy, zz, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contour"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contour(const Tx& x, const Ty& y, const Tz& z, const std::initializer_list<double>& lvl, const std::map<std::string, std::string>& key = {})
    {
        auto xx = eval(x);
        auto yy = eval(y);
        auto zz = eval(z);
        PyObject* args = getContourArgs(xx, yy, zz, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contour"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contour(const Tx& x, const Ty& y, const Tz& z, const std::map<std::string, std::string>& key)
    {
        contour(x, y, z, 10, key);
    }

    void figure()
    {
        load_pyplot();
        PyObject_CallFunctionObjArgs(getPltFun("figure"), NULL);
    }

    void show()
    {
        PyObject_CallFunctionObjArgs(getPltFun("show"), NULL);
    }

    void subplot(int pos, const std::map<std::string, std::string>& key = {})
    {
        PyObject* args = PyTuple_New(1);
        PyTuple_SetItem(args, 0, getPyData(pos));
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("subplot"), args, kwargs);
    }

    void subplot(const std::map<std::string, std::string>& key = {})
    {
        subplot(111, key);
    }

    void xlim(double a, double b)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xlim"), PyFloat_FromDouble(a), PyFloat_FromDouble(b), NULL);
    }

    void ylim(double a, double b)
    {
        PyObject_CallFunctionObjArgs(getPltFun("ylim"), PyFloat_FromDouble(a), PyFloat_FromDouble(b), NULL);
    }

    void title(const std::string& str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("title"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void suptitle(const std::string& str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("suptitle"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void xlabel(const std::string& str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xlabel"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void ylabel(const std::string& str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("ylabel"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void xscale(const std::string& str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xscale"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void yscale(const std::string& str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("yscale"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void legend()
    {
        PyObject_CallFunctionObjArgs(getPltFun("legend"), NULL);
    }

    void colorbar()
    {
        PyObject_CallFunctionObjArgs(getPltFun("colorbar"), NULL);
    }
}; // namespace plt