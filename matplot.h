#pragma once
#include <eigen3/Eigen/Dense>
#include <python/Python.h>

namespace plt
{
    static bool does_py_init = 0;
    static bool does_py_stop = 0;
    static bool does_load_pyplot;

    static PyObject *pModule;
    static PyObject *pFigure;
    static PyObject *pPlot;
    static PyObject *pSubplot;
    static PyObject *pShow;
    static PyObject *pXlim;
    static PyObject *pYlim;

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
            pModule = PyImport_Import(PyUnicode_FromString("matplotlib.pyplot"));
            pFigure = PyObject_GetAttrString(pModule, "figure");
            pPlot = PyObject_GetAttrString(pModule, "plot");
            pSubplot = PyObject_GetAttrString(pModule, "subplot");
            pShow = PyObject_GetAttrString(pModule, "show");
            pXlim = PyObject_GetAttrString(pModule, "xlim");
            pYlim = PyObject_GetAttrString(pModule, "ylim");
            does_load_pyplot = 1;
        }
    }

    void figure()
    {
        load_pyplot();
        PyObject_CallFunctionObjArgs(pFigure, NULL);
    }

    void show()
    {
        PyObject_CallFunctionObjArgs(pShow, NULL);
    }

    void setVal(PyObject *pV, const Eigen::VectorXd &v)
    {
        int n = v.size();
        for (int i = 0; i < n; ++i)
            PyList_SetItem(pV, i, PyFloat_FromDouble(v(i)));
    }

    void plot(const double &x, const double &y)
    {
        PyObject_CallFunctionObjArgs(pPlot, PyFloat_FromDouble(x), PyFloat_FromDouble(y), NULL);
    }

    void plot(const double &x, const double &y, const char *argv)
    {
        PyObject_CallFunctionObjArgs(pPlot, PyFloat_FromDouble(x), PyFloat_FromDouble(y), PyUnicode_FromString(argv), NULL);
    }

    void plot(const Eigen::VectorXd &x, const Eigen::VectorXd &y)
    {
        int n = x.size();
        static PyObject *px = PyList_New(n);
        static PyObject *py = PyList_New(n);
        setVal(px, x);
        setVal(py, y);
        PyObject_CallFunctionObjArgs(pPlot, px, py, NULL);
    }

    void plot(const Eigen::VectorXd &x, const Eigen::VectorXd &y, const char *argv)
    {
        int n = x.size();
        static PyObject *px = PyList_New(n);
        static PyObject *py = PyList_New(n);
        setVal(px, x);
        setVal(py, y);
        PyObject_CallFunctionObjArgs(pPlot, px, py, PyUnicode_FromString(argv), NULL);
    }

    void xlim(int a, int b)
    {
        PyObject_CallFunctionObjArgs(pXlim, PyLong_FromLong(a), PyLong_FromLong(b), NULL);
    }

    void ylim(int a, int b)
    {
        PyObject_CallFunctionObjArgs(pYlim, PyLong_FromLong(a), PyLong_FromLong(b), NULL);
    }

} // namespace plt