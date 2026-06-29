package com.vulnseer.fuzz.runner;

@FunctionalInterface
public interface MethodInvokeTask {

    void invokeTask() throws Exception;

}
