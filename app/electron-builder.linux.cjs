// Use the shared desktop payload with a distinct Debian package and desktop identity.
const fs = require('node:fs')
const path = require('node:path')
const yaml = require('js-yaml')

const config = yaml.load(fs.readFileSync(path.join(__dirname, 'electron-builder.yml'), 'utf8'))
const name = 'ltx-desktop-spacespeak'
const productName = 'LTX Desktop - SpaceSpeak'
config.appId = 'com.spacespeak.ltx-desktop-community'
config.productName = productName
config.extraMetadata = { name, productName }
config.protocols = []
config.publish = null
config.linux.executableName = name
config.linux.artifactName = 'LTX-Desktop-SpaceSpeak-${version}-${arch}.${ext}'
config.deb.packageName = name
config.deb.maintainer = 'SpaceSpeak.com community contributors'
for (const resource of config.extraResources) {
  if (resource.from === 'python-deps-hash.txt') {
    resource.from = 'resources/linux/runtime-deps-sha256.txt'
  }
  if (resource.from === 'backend') {
    resource.filter.unshift('**/*.json')
  }
}
module.exports = config
