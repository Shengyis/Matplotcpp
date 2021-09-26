#pragma once
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#ifndef EIGEN_MATRIXBASE_PLUGIN
#define EIGEN_MATRIXBASE_PLUGIN <MathTools/MatrixBaseAddons.h>
#endif
#include <eigen3/Eigen/Core>
#include <string>
#include <initializer_list>
#include <tuple>
#include <map>
#include <Python.h>
#include <numpy/arrayobject.h>

namespace plt
{
    static bool does_py_init = 0;
    static bool does_py_stop = 0;
    static bool does_load_pyplot = 0;
    static PyObject* plt;
    static PyObject* np;

    template <typename Tx, typename Ty>
    auto meshgrid(const Eigen::MatrixBase<Tx>& x, const Eigen::MatrixBase<Ty>& y)
    {
        int cols = x.size();
        int rows = y.size();
        Eigen::MatrixXd X(rows, cols), Y(rows, cols);
        for (int k = 0; k < rows; ++k)
            X.row(k) = x.transpose();
        for (int k = 0; k < cols; ++k)
            Y.col(k) = y;
        return std::make_tuple(X, Y);
    }

    template <typename T>
    auto meshgrid(const Eigen::MatrixBase<T>& x)
    {
        return meshgrid(x, x);
    }

    void py_init()
    {
        if (!does_py_init)
        {
            Py_Initialize();
            does_py_init = 1;
        }
    }

    void load_pyplot()
    {
        py_init();
        if (!does_load_pyplot)
        {
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
    PyObject* getList(const T& v)
    {
        size_t n = v.size();
        PyObject* pList = PyList_New(n);
        for (int i = 0; i < n; ++i)
            PyList_SetItem(pList, i, PyFloat_FromDouble(double(v[i])));
        return pList;
    }

    template <typename T>
    PyObject* getList(const std::initializer_list<T>& v)
    {
        size_t n = v.size();
        PyObject* pList = PyList_New(n);
        for (int i = 0; i < n; ++i)
            PyList_SetItem(pList, i, PyFloat_FromDouble(double(v.begin()[i])));
        return pList;
    }

    template <>
    PyObject* getList(const double& v)
    {
        PyObject* pList = PyFloat_FromDouble(v);
        return pList;
    }

    template <>
    PyObject* getList(const int& v)
    {
        PyObject* pList = PyLong_FromLong(v);
        return pList;
    }

    template <typename T>
    PyObject* getArray(const T& v)
    {
        npy_intp rows = v.rows();
        npy_intp cols = v.cols();
        npy_intp dim[2] = { cols, rows };
        int nd = 2;
        PyObject* pArray = PyArray_SimpleNewFromData(nd, dim, NPY_FLOAT64, const_cast<double*>(v.data()));
        std::cout << PyArray_DIM(reinterpret_cast<PyArrayObject*>(pArray), 0) << std::endl;
        return pArray;
    }

    template <typename Tx, typename Ty>
    PyObject* getPlotArgs(const Tx& x, const Ty& y, const std::string& str = "")
    {
        PyObject* pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, getList(x));
        PyTuple_SetItem(pArgs, 1, getList(y));
        PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(str.c_str()));
        return pArgs;
    }

    template <typename Tx, typename Ty, typename Tz, typename Tlvl = int>
    PyObject* getContourArgs(const Tx& x, const Ty& y, const Tz& z, const Tlvl& lvl = 10)
    {
        PyObject* pArgs = PyTuple_New(4);
        PyTuple_SetItem(pArgs, 0, getArray(x));
        PyTuple_SetItem(pArgs, 1, getArray(y));
        PyTuple_SetItem(pArgs, 2, getArray(z));
        PyTuple_SetItem(pArgs, 3, getList(lvl));
        return pArgs;
    }

    PyObject* getKwargs(const std::map<std::string, std::string>& key)
    {
        PyObject* pKwargs = PyDict_New();
        for (auto const& item : key)
            PyDict_SetItem(pKwargs, PyUnicode_FromString(item.first.c_str()), PyUnicode_FromString(item.second.c_str()));
        return pKwargs;
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

    template <typename Tx, typename Ty>
    void plot(const Tx& x, const Ty& y, const std::string str = "", const std::map<std::string, std::string>& key = {})
    {
        PyObject* args = getPlotArgs(x, y, str);
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
        PyObject* args = getContourArgs(x, y, z, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contourf"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contourf(const Tx& x, const Ty& y, const Tz& z, const std::initializer_list<double>& lvl, const std::map<std::string, std::string>& key = {})
    {
        PyObject* args = getContourArgs(x, y, z, lvl);
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
        PyObject* args = getContourArgs(x, y, z, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contour"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contour(const Tx& x, const Ty& y, const Tz& z, const std::initializer_list<double>& lvl, const std::map<std::string, std::string>& key = {})
    {
        PyObject* args = getContourArgs(x, y, z, lvl);
        PyObject* kwargs = getKwargs(key);
        PyObject_Call(getPltFun("contour"), args, kwargs);
    }

    template <typename Tx, typename Ty, typename Tz>
    void contour(const Tx& x, const Ty& y, const Tz& z, const std::map<std::string, std::string>& key)
    {
        contour(x, y, z, 10, key);
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