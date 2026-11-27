import { readFileSync, writeFileSync } from "node:fs";
import { NtExecutable, NtExecutableResource, Resource } from "resedit";

const [input, output, name, release] = process.argv.slice(2);
const version = /^(\d+)\.(\d+)\.(\d+)-(?:amd|community)\.(\d+)$/.exec(release);
if (!version || !name) throw new Error("Expected a branch release version and a product name");
const numericVersion = version.slice(1).map(Number);
if (numericVersion.some(n => n > 65535)) throw new Error("Version exceeds Windows VERSIONINFO limits");

// Editing Windows resources creates a new unsigned app executable from Electron.
const exe = NtExecutable.from(readFileSync(input), { ignoreCert: true });
const resources = NtExecutableResource.from(exe);
const versions = Resource.VersionInfo.fromEntries(resources.entries);
if (!versions.length) throw new Error("Electron has no VERSIONINFO resource");
for (const item of versions) {
  for (const language of item.getAllLanguagesForStringValues()) {
    item.setFileVersion(...numericVersion, language.lang);
    item.setProductVersion(...numericVersion, language.lang);
    item.setStringValues(language, {
      ProductName: name,
      FileDescription: name,
      InternalName: name,
      OriginalFilename: name + ".exe",
    });
  }
  item.outputToResourceEntries(resources.entries);
}
resources.outputResource(exe);
writeFileSync(output, Buffer.from(exe.generate()));
