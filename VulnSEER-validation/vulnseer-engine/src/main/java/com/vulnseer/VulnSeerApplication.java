package com.vulnseer;

import cn.hutool.core.io.FileUtil;
import com.vulnseer.config.GlobalConfiguration;
import com.vulnseer.config.ProjectContext;
import lombok.extern.slf4j.Slf4j;
import org.apache.commons.cli.*;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;
import org.springframework.scheduling.annotation.EnableScheduling;
import org.springframework.stereotype.Component;

import java.io.File;
import java.util.Arrays;

import static com.vulnseer.VulnSeer.*;

@Slf4j
@Component
@EnableScheduling
@SpringBootApplication
public class VulnSeerApplication {

    private static VulnSeer vulnseer;
    private static File inputJsonFile;

    @Autowired
    public void setClientDefender(VulnSeer vulnseer) {
        VulnSeerApplication.vulnseer = vulnseer;
    }

    public static void main(String[] args) throws Exception {
        SpringApplication.run(VulnSeerApplication.class, args);

        log.info("input args: {}", Arrays.toString(args));
        ProjectContext projectContext = parseArgs(args);

        vulnseer.run(projectContext, inputJsonFile);
    }

    private static ProjectContext parseArgs(String[] args) {
        Options options = new Options();
        options.addOption(new Option(PROJECT_ROOT_PATH, true, "detect project root path"));
        options.addOption(new Option(SKIP_FUZZING, false, "whether to skip the fuzzing procedure"));
        options.addOption(new Option(OUTPUT_DIR, true, "result output directory"));
        options.addOption(new Option(TARGET_VULNERABILITY, true, "target detect vulnerability"));
        options.addOption("in", INPUT_FILE, true, "unified validation input json file path");

        CommandLineParser parser = new DefaultParser();
        ProjectContext projectContext = null;

        try {
            CommandLine line = parser.parse(options, args);
            String projectPath = line.getOptionValue(PROJECT_ROOT_PATH);

            if (projectPath == null) {
                log.warn("project path is null, start checking default demo project");
                projectPath = System.getProperty("user.dir") + File.separator + "test-tool-demo";
            }

            File projectDir = FileUtil.file(projectPath);
            if (projectDir == null || !projectDir.exists() || !projectDir.isDirectory()) {
                throw new RuntimeException("the input project directory must be exist");
            }

            String outputDir = line.getOptionValue(OUTPUT_DIR);
            if (outputDir != null) {
                File outputFile = FileUtil.file(outputDir);
                GlobalConfiguration.OUTPUT_DIR_PATH = outputFile.getAbsolutePath();
                GlobalConfiguration.OUTPUT_DIR = outputFile;
            }

            String targetVulnerability = line.getOptionValue(TARGET_VULNERABILITY);
            if (targetVulnerability != null) {
                GlobalConfiguration.TARGET_VULNERABILITY = targetVulnerability;
            }

            String inputPath = line.getOptionValue(INPUT_FILE);
            if (inputPath == null) {
                throw new RuntimeException("missing required argument: --" + INPUT_FILE);
            }

            inputJsonFile = FileUtil.file(inputPath);
            if (!inputJsonFile.exists() || !inputJsonFile.isFile()) {
                throw new RuntimeException("invalid input json file: " + inputPath);
            }

            projectContext = new ProjectContext(projectPath, projectDir, line);
        } catch (ParseException e) {
            log.error("ParsingCommandError: ", e);
        }

        return projectContext;
    }
}