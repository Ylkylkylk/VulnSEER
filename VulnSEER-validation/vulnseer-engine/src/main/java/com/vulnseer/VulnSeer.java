package com.vulnseer;

import cn.hutool.core.date.DateUtil;
import cn.hutool.core.date.TimeInterval;
import cn.hutool.core.io.FileUtil;
import com.vulnseer.config.ClientProjectProperty;
import com.vulnseer.config.ConfigProperty;
import com.vulnseer.config.GlobalClassLoader;
import com.vulnseer.config.GlobalConfiguration;
import com.vulnseer.config.ProjectContext;
import com.vulnseer.dependency.MavenDependencyTree;
import com.vulnseer.fuzz.FuzzClassLoader;
import com.vulnseer.service.PoCValidationService;
import com.vulnseer.service.ValidationReport;
import com.vulnseer.util.TimeUtil;
import com.vulnseer.util.URLUtil;
import lombok.extern.slf4j.Slf4j;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.context.annotation.ComponentScan;
import org.springframework.context.annotation.Import;
import org.springframework.stereotype.Component;

import java.io.File;
import java.net.MalformedURLException;
import java.net.URLClassLoader;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;
import java.util.Map;
import java.util.TreeMap;
import java.util.stream.Collectors;

@Slf4j
@Import(cn.hutool.extra.spring.SpringUtil.class)
@Component
@ComponentScan(basePackages = {"cn.hutool.extra.spring"})
public class VulnSeer {

    @Autowired
    ConfigProperty configProperty;

    public static final String PROJECT_ROOT_PATH = "p";
    public static final String SKIP_FUZZING = "skipFuzz";
    public static final String TARGET_VULNERABILITY = "vul";
    public static final String OUTPUT_DIR = "output";
    public static final String INPUT_FILE = "input";

    private void prepareOutputDir() {
        FileUtil.del(GlobalConfiguration.OUTPUT_DIR);
        FileUtil.mkdir(GlobalConfiguration.OUTPUT_DIR);
    }

    private void prepareCacheDir() {
        FileUtil.del(GlobalConfiguration.CACHE_DIR_PATH);
        FileUtil.mkdir(GlobalConfiguration.CACHE_DIR_PATH);
        FileUtil.mkdir(GlobalConfiguration.JAR_DIR_PATH);
    }

    private MavenDependencyTree generateDependencyTree(ProjectContext projectContext) {
        return TimeUtil.runTask(() -> {
            try {
                MavenDependencyTree dependencyTree = new MavenDependencyTree(projectContext.getProjectDir());
                projectContext.setDependencyTree(dependencyTree);
                return dependencyTree;
            } catch (Exception e) {
                log.error("ParsingMavenDependencyTreeError: ", e);
                throw new RuntimeException(e);
            }
        }, "generate-dependency-tree");
    }

    public void initGlobalClassLoader(ProjectContext projectContext) {
        TimeUtil.runTask(() -> {
            MavenDependencyTree dependencyTree = projectContext.getDependencyTree();

            List<String> dependencyJarList = Arrays.stream(FileUtil.ls(GlobalConfiguration.JAR_DIR_PATH))
                    .filter(File::isFile)
                    .map(File::getAbsolutePath)
                    .filter(path -> path.endsWith(".jar"))
                    .collect(Collectors.toList());

            String[] jarPaths = new String[1 + dependencyJarList.size()];
            jarPaths[0] = dependencyTree.getJarFile().getAbsolutePath();

            for (int i = 0; i < dependencyJarList.size(); i++) {
                jarPaths[i + 1] = dependencyJarList.get(i);
            }

            try {
                GlobalClassLoader.init(jarPaths);
            } catch (MalformedURLException e) {
                log.error("GlobalClassLoaderInitError: ", e);
                throw new RuntimeException(e);
            }

            Thread.currentThread().setContextClassLoader(GlobalClassLoader.getInstance().getClassLoader());
        }, "init-global-classloader");
    }

    /**
     * Keep the original VulnSeer instrumented-loader initialization path
     * so STATE_TABLE instrumentation checks remain available.
     */
    private void initializeExecutionEnv(ProjectContext projectContext) {
        TimeUtil.runTask(() -> {
            MavenDependencyTree dependencyTree = projectContext.getDependencyTree();

            List<String> jarPathList = new ArrayList<>(dependencyTree.getDependencyJarPathList());
            jarPathList.add(dependencyTree.getJarFile().getAbsolutePath());

            ClassLoader instrumentedLoader;
            ClassLoader normalLoader;

            try {
                instrumentedLoader = new FuzzClassLoader(jarPathList);
                normalLoader = new URLClassLoader(
                        URLUtil.stringsToUrls(jarPathList),
                        ClassLoader.getSystemClassLoader().getParent()
                );
            } catch (MalformedURLException e) {
                log.error("InitializeExecutionEnvError: ", e);
                throw new RuntimeException(e);
            }

            ClientProjectProperty.setFuzzLoader(normalLoader);
            ClientProjectProperty.setInstrumentedFuzzLoader(instrumentedLoader);

            projectContext.setInstrumentedFuzzClassLoader(instrumentedLoader);
            projectContext.setFuzzClassLoader(normalLoader);
        }, "initialize-execution-environment");
    }

    private List<ValidationReport> validateInput(ProjectContext projectContext, File inputJsonFile) {
        return TimeUtil.runTask(() -> {
            PoCValidationService validationService = new PoCValidationService(projectContext, configProperty);
            return validationService.validateAll(inputJsonFile);
        }, "validate-input");
    }

    private void generateValidationReport(ProjectContext projectContext, List<ValidationReport> reports) {
        TimeUtil.runTask(() -> {
            PoCValidationService validationService = new PoCValidationService(projectContext, configProperty);
            validationService.generateValidationReport(reports);
        }, "generate-validation-report");
    }

    private String buildDepthStatsSummary(List<ValidationReport> reports) {
        Map<Integer, long[]> depthStatsMap = new TreeMap<>();

        for (ValidationReport report : reports) {
            int depth = 0;
            if (report.getCallChain() != null && !report.getCallChain().isEmpty()) {
                depth = Math.max(0, report.getCallChain().size() - 1);
            }

            long[] stats = depthStatsMap.computeIfAbsent(depth, key -> new long[3]);
            stats[0]++;
            if (report.isSinkReached()) {
                stats[1]++;
            }
            if (report.isVulnerabilityTriggered()) {
                stats[2]++;
            }
        }

        return depthStatsMap.entrySet().stream()
                .map(entry -> String.format("depth=%d(total=%d, reached=%d, triggered=%d)",
                        entry.getKey(), entry.getValue()[0], entry.getValue()[1], entry.getValue()[2]))
                .collect(Collectors.joining("; "));
    }

    public void work(ProjectContext projectContext, File inputJsonFile) {
        TimeUtil.runTask(() -> {
            TimeInterval timer = DateUtil.timer();
            timer.start("total-task");
            timer.start("analysis-task");

            generateDependencyTree(projectContext);
            initGlobalClassLoader(projectContext);
            initializeExecutionEnv(projectContext);

            long analysisTime = timer.interval("analysis-task");
            projectContext.setAnalysisTime(analysisTime);

            timer.start("fuzz-task");
            List<ValidationReport> reports = validateInput(projectContext, inputJsonFile);
            long fuzzTime = timer.interval("fuzz-task");
            projectContext.setFuzzTime(fuzzTime);

            long totalTime = timer.interval("total-task");
            projectContext.setTotalTime(totalTime);

            for (ValidationReport report : reports) {
                report.setAnalysisTime(analysisTime);
                report.setFuzzTime(fuzzTime);
                report.setTotalTime(totalTime);
            }

            generateValidationReport(projectContext, reports);

            long reachedCount = reports.stream().filter(ValidationReport::isSinkReached).count();
            long triggeredCount = reports.stream().filter(ValidationReport::isVulnerabilityTriggered).count();
            String depthStatsSummary = buildDepthStatsSummary(reports);

            log.info("validation finished. total = {}, triggered = {}, reached = {}, depthStats = [{}]",
                    reports.size(), triggeredCount, reachedCount, depthStatsSummary);
        }, "entire-work");
    }

    public void workSingleChain(ProjectContext projectContext, File inputJsonFile) {
        work(projectContext, inputJsonFile);
    }

    public void run(ProjectContext projectContext, File inputJsonFile) {
        if (projectContext == null) {
            throw new RuntimeException("project context can't be null");
        }
        if (inputJsonFile == null || !inputJsonFile.exists()) {
            throw new RuntimeException("input json file not found: " + inputJsonFile);
        }

        log.info("the work directory is {}", GlobalConfiguration.WORK_DIR.getAbsolutePath());
        log.info("the cache directory is {}", GlobalConfiguration.CACHE_DIR.getAbsolutePath());
        log.info("the output directory is {}", GlobalConfiguration.OUTPUT_DIR.getAbsolutePath());
        log.info("input json file = {}", inputJsonFile.getAbsolutePath());

        prepareCacheDir();
        prepareOutputDir();

        work(projectContext, inputJsonFile);
    }

    public void singleRun(ProjectContext projectContext, File inputJsonFile) {
        run(projectContext, inputJsonFile);
    }
}
