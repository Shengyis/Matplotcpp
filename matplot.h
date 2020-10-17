#pragma once
#include <string>
#include <map>
#include <type_traits>
#include <python/Python.h>

namespace plt
{
    static bool does_py_init = 0;
    static bool does_py_stop = 0;
    static bool does_load_pyplot = 0;
    static PyObject *pModule;

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
            pModule = PyImport_ImportModule("matplotlib.pyplot");
            does_load_pyplot = 1;
        }
    }

    PyObject *getPltFun(const char *argv)
    {
        return PyObject_GetAttrString(pModule, argv);
    }

    template <typename T>
    PyObject *getVar(const T &v)
    {
        size_t n = v.size();
        PyObject *pV = PyList_New(n);
        for (int i = 0; i < n; ++i)
            PyList_SetItem(pV, i, PyFloat_FromDouble(v[i]));
        return pV;
    }

    template <>
    PyObject *getVar(const double &v)
    {
        PyObject *pV = PyList_New(1);
        PyList_SetItem(pV, 0, PyFloat_FromDouble(v));
        return pV;
    }

    template <typename Tx, typename Ty>
    PyObject *getArgs(const Tx &x, const Ty &y, const std::string &str = "")
    {
        PyObject *pArgs = PyTuple_New(3);
        PyTuple_SetItem(pArgs, 0, getVar(x));
        PyTuple_SetItem(pArgs, 1, getVar(y));
        PyTuple_SetItem(pArgs, 2, PyUnicode_FromString(str.c_str()));
        return pArgs;
    }

    PyObject *getKwargs(const std::map<std::string, std::string> &key)
    {
        PyObject *pKwargs = PyDict_New();
        for (auto const &item : key)
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
    void plot(const Tx &x, const Ty &y, const std::string str = "", const std::map<std::string, std::string> &key = {})
    {
        PyObject *args = getArgs(x, y, str);
        PyObject *kwargs = getKwargs(key);
        PyObject_Call(getPltFun("plot"), args, kwargs);
    }

    template <typename Tx, typename Ty>
    void plot(const Tx &x, const Ty &y, const std::map<std::string, std::string> &key = {})
    {
        plot(x, y, "", key);
    }

    void xlim(double a, double b)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xlim"), PyFloat_FromDouble(a), PyFloat_FromDouble(b), NULL);
    }

    void ylim(double a, double b)
    {
        PyObject_CallFunctionObjArgs(getPltFun("ylim"), PyFloat_FromDouble(a), PyFloat_FromDouble(b), NULL);
    }

    void title(const std::string &str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("title"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void xlabel(const std::string &str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("xlabel"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void ylabel(const std::string &str)
    {
        PyObject_CallFunctionObjArgs(getPltFun("ylabel"), PyUnicode_FromString(str.c_str()), NULL);
    }

    void legend()
    {
        PyObject_CallFunctionObjArgs(getPltFun("legend"), NULL);
    }
} // namespace plt