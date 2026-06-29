package com.vulnseer.config;

import com.vulnseer.dependency.MavenDependencyTree;
import org.apache.commons.cli.CommandLine;

import java.io.File;

public class ProjectContext {

    private Long totalTime;

    private Long analysisTime;

    private Long fuzzTime;

    private String projectPath;

    private File projectDir;

    private CommandLine commandLine;

    private ClassLoader instrumentedFuzzClassLoader;

    private ClassLoader fuzzClassLoader;

    private MavenDependencyTree dependencyTree;

    public ProjectContext() {

    }

    public ProjectContext(String projectPath, File projectDir, CommandLine commandLine) {
        this.projectPath = projectPath;
        this.projectDir = projectDir;
        this.commandLine = commandLine;
    }

    public void setFuzzClassLoader(ClassLoader fuzzClassLoader) {
        this.fuzzClassLoader = fuzzClassLoader;
    }

    public ClassLoader getFuzzClassLoader() {
        return fuzzClassLoader;
    }

    public ClassLoader getInstrumentedFuzzClassLoader() {
        return instrumentedFuzzClassLoader;
    }

    public void setInstrumentedFuzzClassLoader(ClassLoader instrumentedFuzzClassLoader) {
        this.instrumentedFuzzClassLoader = instrumentedFuzzClassLoader;
    }

    public CommandLine getCommandLine() {
        return commandLine;
    }

    public void setCommandLine(CommandLine commandLine) {
        this.commandLine = commandLine;
    }

    public File getProjectDir() {
        return projectDir;
    }

    public void setProjectDir(File projectDir) {
        this.projectDir = projectDir;
    }

    public String getProjectPath() {
        return projectPath;
    }

    public void setProjectPath(String projectPath) {
        this.projectPath = projectPath;
    }

    public MavenDependencyTree getDependencyTree() {
        return dependencyTree;
    }

    public void setDependencyTree(MavenDependencyTree dependencyTree) {
        this.dependencyTree = dependencyTree;
    }

    public Long getAnalysisTime() {
        return analysisTime;
    }

    public Long getFuzzTime() {
        return fuzzTime;
    }

    public Long getTotalTime() {
        return totalTime;
    }

    public void setAnalysisTime(Long analysisTime) {
        this.analysisTime = analysisTime;
    }

    public void setFuzzTime(Long fuzzTime) {
        this.fuzzTime = fuzzTime;
    }

    public void setTotalTime(Long totalTime) {
        this.totalTime = totalTime;
    }
}
